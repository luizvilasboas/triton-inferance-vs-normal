from ultralytics import YOLO
import subprocess
import time
import contextlib
from tritonclient.http import InferenceServerClient
from pathlib import Path
import cv2
from utils import get_proc_filename
from abc import ABC, abstractmethod


class YOLOv8TestType:
    VIDEO = 0
    IMAGE = 1


class YOLOv8:
    def __ini__(self, type: YOLOv8TestType, data: str) -> None:
        self.type = type
        self.data = data
        self.model = None

    @abstractmethod
    def run(self) -> None:
        pass

    def _run_model_image(self):
        results = self.model(self.data, verbose=False)

        results[0].save(get_proc_filename(self.data))

    def _run_model_video(self) -> None:
        cap = cv2.VideoCapture(self.data)
        out = cv2.VideoWriter(get_proc_filename(self.data), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS), (int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while cap.isOpened():
            success, frame = cap.read()

            if success:
                results = self.model.track(frame, persist=True, verbose=False)

                annotated_frame = results[0].plot()

                out.write(annotated_frame)
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()


class NormalYOLOv8(YOLOv8):
    def __init__(self, type: YOLOv8TestType, data: str) -> None:
        self.type = type
        self.model = YOLO("yolov8n.pt", task='detect')
        self.data = data

    def run(self) -> None:
        if self.type == YOLOv8TestType.IMAGE:
            self._run_model_image()
        elif self.type == YOLOv8TestType.VIDEO:
            self._run_model_video()


class YOLOv8Triton(YOLOv8):
    def __init__(self, type: YOLOv8TestType, data: str, setup=False) -> None:
        self.type = type
        self.setup = setup
        self.model = YOLO(f'http://localhost:8000/yolo', task='detect')
        self.data = data

    def run(self) -> None:
        triton_repo_path = "tmp/triton_repo"

        if self.setup:
            triton_repo_path = self.__setup_repository()

        container_id = self.__init_inferance_server(triton_repo_path)

        if self.type == YOLOv8TestType.IMAGE:
            self._run_model_image()
        elif self.type == YOLOv8TestType.VIDEO:
            self._run_model_video()

        self.__end_inferance_server(container_id)

    def __setup_repository(self) -> str:
        model = YOLO('yolov8n.pt')
        onnx_file = model.export(format='onnx', dynamic=True)

        triton_repo_path = Path('tmp') / 'triton_repo'
        triton_model_path = triton_repo_path / 'yolo'

        (triton_model_path / '1').mkdir(parents=True, exist_ok=True)

        Path(onnx_file).rename(triton_model_path / '1' / 'model.onnx')

        (triton_model_path / 'config.pbtxt').touch()

        return triton_repo_path

    def __init_inferance_server(self, triton_repo_path: str) -> str:
        tag = 'nvcr.io/nvidia/tritonserver:23.09-py3'
        subprocess.call(f'docker pull {tag}', shell=True)
        container_id = subprocess.check_output(
            f'docker run -d --rm -v ./{triton_repo_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models', shell=True).decode('utf-8').strip()

        triton_client = InferenceServerClient(
            url='localhost:8000', verbose=False, ssl=False)

        for _ in range(10):
            with contextlib.suppress(Exception):
                assert triton_client.is_model_ready("yolo")
                break

            time.sleep(1)

        return container_id

    def __end_inferance_server(self, container_id: str) -> None:
        subprocess.call(f'docker kill {container_id}', shell=True)


if __name__ == '__main__':
    triton_yolov8 = YOLOv8Triton(
        YOLOv8TestType.IMAGE, "images/dog-test-image.jpg")
    triton_yolov8.run()
