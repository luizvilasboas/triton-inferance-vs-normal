from ultralytics import YOLO
import time
import cv2
import argparse
from utils import get_proc_filename, calculate_time, add_prefix_filename
from abc import abstractmethod
import time


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
        out = cv2.VideoWriter(add_prefix_filename(get_proc_filename(self.data), self.prefix), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        frame_numbers = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        # print(f"> Total de frames: {frame_numbers}")
        while cap.isOpened():
            success, frame = cap.read()

            # print(f"Frame atual: {frame_count}", end="\r")

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
    def __init__(self, type: YOLOv8TestType, data: str, setup=False, model_url = "http://localhost:8000/yolo") -> None:
        self.type = type
        self.setup = setup
        self.model = YOLO(model_url, task='detect')
        self.data = data
        self.prefix = "triton-"

    def run(self) -> tuple[int, int, int]:
        print("> Inicializando a inferência com o triton")

        start = time.time()
        if self.type == YOLOv8TestType.IMAGE:
            self._run_model_image()
        elif self.type == YOLOv8TestType.VIDEO:
            self._run_model_video()
        end = time.time()

        time_total = end - start 
        hours, minutes, seconds = calculate_time(time_total)

        return hours, minutes, seconds

def main() -> None:
    parser = argparse.ArgumentParser(description="Simple program to test if trition is good for our use")
    parser.add_argument('data', metavar='DATA', type=str, help="The name of the file (image or video) to use")
    parser.add_argument("--image", "-i", action="store_true", help="Set data as a image")
    parser.add_argument("--video", "-v", action="store_true", help="Set data as a video")
    parser.add_argument("--setup", "-s", action="store_true", help="Setup triton repository with YOLOv8")
    parser.add_argument("--triton", "-t", action="store_true", help="Just use triton")
    parser.add_argument("--normal", "-n", action="store_true", help="Just use normal")

    args = parser.parse_args()
    data = args.data
    setup = args.setup
    triton = args.triton
    normal = args.normal

    triton_yolov8 = normal_yolov8 = None

    model_url="http://localhost:8000/yolo"
    model_file="yolov8n.onnx"

    if args.image:
        if triton:
            triton_yolov8 = YOLOv8Triton(YOLOv8TestType.IMAGE, data, setup, model_url)
        elif normal:
            normal_yolov8 = NormalYOLOv8(YOLOv8TestType.IMAGE, data, model_file)
    elif args.video:
        if triton:
            triton_yolov8 = YOLOv8Triton(YOLOv8TestType.VIDEO, data, setup, model_url)
        elif normal:
            normal_yolov8 = NormalYOLOv8(YOLOv8TestType.VIDEO, data, model_file)
    
    if triton:
        print("> Começando a inferência usando o triton")
        hours, minutes, seconds = triton_yolov8.run()

        print(f"> O tempo total de Inferência usando o triton foi de {hours} horas, {minutes} minutos, {seconds} segundos.")

    if normal:
        print("> Começando a inferência usando a forma normal")
        hours, minutes, seconds = normal_yolov8.run()

        print(f"> O tempo total de Inferência usando da forma normal foi de {hours} horas, {minutes} minutos, {seconds} segundos.")

if __name__ == '__main__':
    main()
