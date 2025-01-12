# Pose Recognition Using Python

## Description

This project implements a human pose recognition system using Python, OpenCV, and MediaPipe. It detects 33 key body landmarks in static images and visualizes the pose skeleton by connecting these landmarks. The system is designed to handle varying body orientations and environments.

## Table of Contents

-   [Installation](#installation)
-   [Usage](#usage)
-   [Project Structure](#project-structure)
-   [Results](#results)
-   [Contributing](#contributing)
-   [References](#references)

## Installation

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/shivamvishwakarma2k2/Pose-Recognition-Using-Python.git
    cd Pose-Recognition-Using-Python
    ```

2. **Create a Virtual Environment**:

    ```bash
    python -m venv venv
    ```

3. **Activate the Virtual Environment**:

    ```bash
    venv\Scripts\activate
    ```

4. **Install Required Dependencies**:

    ```bash
    pip install opencv-python mediapipe numpy matplotlib
    ```

## Usage

### Prepare Input Images

Place the images you want to process in the `images` directory within the project folder.

### Run the Pose Recognition Script

Execute the main script to process the images:

```bash
python main.py
```

OR 

> Directly run the cell of main.ipynb file


### Project Structure

```bash
Pose-Recognition-Using-Python/
├── images/
│ ├── image-1.png
│ ├── image-2.png
│ └── ...
├── main.py
├── main.ipynb
├── pose_recognize_and_estimate.py
└── README.md
```

-   **images/**: Directory containing input images.
-   **main.py**: Main script to run the pose recognition.
-   **main.ipynb**: Can run main.py file from this file.
-   **pose_recognize_and_estimate.py**: Contains functions for pose detection and visualization.
-   **README.md**: Project documentation.

## Results

The processed images will display detected body landmarks connected by lines, forming a recognizable pose skeleton. Below is an example of the output:

![Picture2](https://github.com/user-attachments/assets/338c1da5-289f-4669-b7ae-015bd54fdd24)
![Picture1](https://github.com/user-attachments/assets/ce8b992a-780c-4819-968c-028c1feafd4d)


## Contributing

Contributions are welcome! Please fork the repository and use a feature branch. Pull requests are warmly welcome.

## References

-   [MediaPipe Pose Documentation](https://google.github.io/mediapipe/)
-   [OpenCV Documentation](https://docs.opencv.org/)
-   [OpenCV Python Tutorials](https://opencv-python-tutroals.readthedocs.io/)
-   [GeeksforGeeks Pose Recognition Tutorial](https://www.geeksforgeeks.org/)
