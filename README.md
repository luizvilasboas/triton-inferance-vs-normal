# triton-inferance-vs-normal

This project aims to determine the most efficient approach for deploying YOLO-based object detection models in your specific environment. We'll compare two primary methods:

* **Direct Model Loading**: This involves loading the YOLO model directly into your application using a framework-specific library (e.g., PyTorch, TensorFlow).

* **Triton Inference Server**: This approach utilizes NVIDIA's Triton Inference Server as an intermediary, enabling centralized model management, versioning, and potentially improved performance on NVIDIA GPUs.

## Requirements

1. **Docker**: Use Docker version 24.07 or greater.
2. **Python**: Use Python 3 version 3.10 or greater.

## Usage

1. Install the dependencies with the command bellow: 

    ```
    pip install -r requirements.txt
    ```

2. Setup the repository for Triton by running this bash script:

    ```
    bash repository.sh 
    ```

3. Run the inference server by running this bash script:

    ```
    bash server.sh
    ```

4. Run the client:

    ```
    python3 main.py --video videos/family-dog-test-video.mp4 --triton --normal --http
    ```

5. Use more options:


    ```
    python3 main.py --help
    ```

## Contributing

If you wish to contribute to this project, feel free to open a merge request. We welcome all forms of contribution!

## License

This project is licensed under the [The Unlicense](https://gitlab.com/olooeez/triton-inferance-vs-normal/-/blob/main/LICENSE). Refer to the LICENSE file for more details.
