# triton-inference-vs-normal

> A comparative analysis between deploying a YOLO object detection model directly versus serving it via the NVIDIA Triton Inference Server.

## About the Project

This project aims to determine the most efficient approach for deploying YOLO-based object detection models in a specific environment. It compares two primary methods:

1.  **Direct Model Loading**: Loading the YOLO model directly into a Python application using a framework-specific library (e.g., PyTorch).
2.  **Triton Inference Server**: Utilizing NVIDIA's Triton as an intermediary, which enables centralized model management, versioning, and potentially improved performance.

The goal is to benchmark and understand the trade-offs between these two deployment strategies.

## Tech Stack

*   [Python](https://www.python.org/)
*   [YOLO](https://github.com/ultralytics/yolov5)
*   [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server)
*   [Docker](https://www.docker.com/)

## Usage

Below are the instructions for you to set up and run the comparison.

### Prerequisites

You need to have the following software installed:

*   [Python](https://www.python.org/downloads/) (version 3.10 or higher)
*   [Docker](https://docs.docker.com/get-docker/) (version 24.07 or higher)

### Installation and Setup

Follow the steps below:

1.  **Clone the repository**
    ```bash
    git clone https://github.com/luizvilasboas/triton-inferance-vs-normal.git
    ```

2.  **Navigate to the project directory**
    ```bash
    cd triton-inferance-vs-normal
    ```

3.  **Install Python dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up the Triton model repository**
    Run the provided script to prepare the models for Triton.
    ```bash
    bash repository.sh
    ```

### Workflow

1.  **Start the Triton Server**
    Run the server script to launch the Triton container.
    ```bash
    bash server.sh
    ```

2.  **Run the client for comparison**
    Execute the client script, providing a video and flags to select the inference mode(s).
    ```bash
    # Example: Run both Triton and normal inference over HTTP
    python3 main.py --video videos/family-dog-test-video.mp4 --triton --normal --http
    ```
    Use `python3 main.py --help` to see all available options.

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request.

## License

This project is licensed under The Unlicense. See the `LICENSE` file for details.
