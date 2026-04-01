# Hand-Gesture-Recognition

## Overview
The Hand Gesture Recognition Software is an advanced system that detects and interprets hand gestures using computer vision and machine learning techniques. This software enables gesture-based interactions for applications such as gaming, accessibility tools, and smart home controls.

## Features
- **Real-time Gesture Recognition**: Detects and classifies hand gestures in real-time.
- **Machine Learning-based Model**: Utilizes deep learning models for accurate recognition.
- **Multi-platform Support**: Works on Windows, Linux, and macOS.
- **Customizable Gestures**: Allows users to define and train new gestures.
- **Integration with External Applications**: Can be connected with other software via API.

## Requirements
- Python 3.8+
- OpenCV
- TensorFlow/Keras
- Mediapipe
- NumPy
- Matplotlib (for visualization)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/Saanket-Das/Hand-Gesture-Recognition.git
   cd hand-gesture-recognition
   ```
2. Run the software:
   ```sh
   python test.py
   ```

## Usage
1. Start the application.
2. Position your hand within the camera’s view.
3. Perform a predefined gesture.
4. The software will recognize and display the detected gesture.
5. Customize gestures by training the model with new hand poses.

## Training Custom Gestures
To add new gestures:
1. Collect gesture images using `capture.py`.
2. Label the images and store them in the dataset folder.
3. Train the model using `train.py`:
   ```sh
   python train.py
   ```

## API Integration
The software provides an API to integrate with external applications. Example request:
```sh
POST /api/recognize
Content-Type: application/json
{
  "image": "base64-encoded-image-data"
}
```

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests.



