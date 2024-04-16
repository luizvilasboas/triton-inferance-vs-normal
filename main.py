from ultralytics import YOLO
import subprocess
import time
import contextlib
from tritonclient.http import InferenceServerClient
from pathlib import Path
import cv2
import argparse
from utils import get_proc_filename, calculate_time, add_prefix_filename
from abc import abstractmethod
import time
import sys


class YOLOv8TestType:
    VIDEO = 0
    IMAGE = 1


class YOLOv8:
    def __ini__(self, type: YOLOv8TestType, data: str) -> None:
        self.type = type
        self.data = data
        self.model = None
        self.prefix = None

    @abstractmethod
    def run(self) -> tuple[int, int, int]:
        pass

    def _run_model_image(self):
        results = self.model(self.data, verbose=False)

        results[0].save(add_prefix_filename(get_proc_filename(self.data), self.prefix))

    def _run_model_video(self) -> None:
        cap = cv2.VideoCapture(self.data)
        out = cv2.VideoWriter(add_prefix_filename(get_proc_filename(self.data), self.prefix), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), cap.get(cv2.CAP_PROP_FPS), (int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        frame_numbers = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        print(f"> Total de frames: {frame_numbers}")
        while cap.isOpened():
            success, frame = cap.read()

            print(f"Frame atual: {frame_count}", end="\r")

            if success:
                results = self.model.track(frame, persist=True, verbose=False)

                annotated_frame = results[0].plot()

                out.write(annotated_frame)
            else:
                break

            frame_count += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()


class NormalYOLOv8(YOLOv8):
    def __init__(self, type: YOLOv8TestType, data: str, model_file = "yolov8n.pt") -> None:
        self.type = type
        self.model = YOLO(model_file, task='detect')
        self.data = data
        self.prefix = "normal-"

    def run(self) -> tuple[int, int, int]:
        print("> Inicializando a inferência da maneira normal")

        start = time.time()
        if self.type == YOLOv8TestType.IMAGE:
            self._run_model_image()
        elif self.type == YOLOv8TestType.VIDEO:
            self._run_model_video()
        end = time.time()

        time_total = end - start 
        hours, minutes, seconds = calculate_time(time_total)

        return hours, minutes, seconds


class YOLOv8Triton(YOLOv8):
    def __init__(self, type: YOLOv8TestType, data: str, setup=False, model_url = "http://localhost:8000/yolo", model_file = "yolov8n.pt") -> None:
        self.type = type
        self.setup = setup
        self.model = YOLO(model_url, task='detect')
        self.data = data
        self.prefix = "triton-"
        self.model_file = model_file

    def run(self) -> tuple[int, int, int]:
        triton_repo_path = "tmp/triton_repo"

        if self.setup:
            triton_repo_path = self.__setup_repository()

        container_id = self.__init_inferance_server(triton_repo_path)

        print("> Inicializando a inferência com o triton")

        start = time.time()
        if self.type == YOLOv8TestType.IMAGE:
            self._run_model_image()
        elif self.type == YOLOv8TestType.VIDEO:
            self._run_model_video()
        end = time.time()

        time_total = end - start 
        hours, minutes, seconds = calculate_time(time_total)

        self.__end_inferance_server(container_id)

        return hours, minutes, seconds

    def __setup_repository(self) -> str:
        model = YOLO(self.model_file)
        onnx_file = model.export(format='onnx', dynamic=True)

        triton_repo_path = Path('tmp') / 'triton_repo'
        triton_model_path = triton_repo_path / 'yolo'

        (triton_model_path / '1').mkdir(parents=True, exist_ok=True)

        Path(onnx_file).rename(triton_model_path / '1' / 'model.onnx')

        (triton_model_path / 'config.pbtxt').touch()

        return triton_repo_path

    def __init_inferance_server(self, triton_repo_path: str) -> str:
        tag = 'nvcr.io/nvidia/tritonserver:23.09-py3'

        print("> Puxando servidor trition do docker")
        subprocess.call(f'docker pull {tag}', shell=True)

        print("> Inicializando o servidor de inferência do triton")
        container_id = subprocess.check_output(
            f'docker run --gpus all --net=host -d --rm -v ./{triton_repo_path}:/models -p 8000:8000 -p 8001:8001 -p 8002:8002 {tag} tritonserver --model-repository=/models', shell=True).decode('utf-8').strip()

        triton_client = InferenceServerClient(
            url='localhost:8000', verbose=False, ssl=False)

        for _ in range(10):
            with contextlib.suppress(Exception):
                assert triton_client.is_model_ready("yolo")
                break

            time.sleep(1)

        return container_id

    def __end_inferance_server(self, container_id: str) -> None:
        print("> Matando o servidor de inferência do triton")
        subprocess.call(f'docker kill {container_id}', shell=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple program to test if trition is good for our use")
    parser.add_argument('data', metavar='DATA', type=str, help="The name of the file (image or video) to use")
    parser.add_argument("--image", "-i", action="store_true", help="Set data as a image")
    parser.add_argument("--video", "-v", action="store_true", help="Set data as a video")
    parser.add_argument("--setup", "-s", action="store_true", help="Setup triton repository with YOLOv8")

    args = parser.parse_args()
    data = args.data
    setup = args.setup

    triton_yolov8 = normal_yolov8 = None

    model_url="http://localhost:8000/yolo"
    model_file="yolov8n.pt"

    if args.image:
        triton_yolov8 = YOLOv8Triton(YOLOv8TestType.IMAGE, data, setup, model_url, model_file)
        normal_yolov8 = NormalYOLOv8(YOLOv8TestType.IMAGE, data, model_file)
    elif args.video:
        triton_yolov8 = YOLOv8Triton(YOLOv8TestType.VIDEO, data, setup, model_url, model_file)
        normal_yolov8 = NormalYOLOv8(YOLOv8TestType.VIDEO, data, model_file)
    
    print("> Começando a inferência usando o triton")
    hours, minutes, seconds = triton_yolov8.run()

    print(f"> O tempo total de Inferência usando o triton foi de {hours} horas, {minutes} minutos, {seconds} segundos.")

    print("> Começando a inferência usando a forma normal")
    hours, minutes, seconds = normal_yolov8.run()

    print(f"> O tempo total de Inferência usando da forma normal foi de {hours} horas, {minutes} minutos, {seconds} segundos.")

if __name__ == '__main__':
    main()
