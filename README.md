# triton-inferance-vs-normal

Basic project to test if is better to use triton inferance server or just load YOLO based models normaly.

## Requirements

1. Docker
2. Python

## Usage

1. Install dependencies:

```
pip install -r Requirements.txt
```

2. Setup the repository and create inferance into a image or video:

```
python3 main.py --image --setup path/to/image.png
# or
python3 main.py --video --setup path/to/video.mp4
```