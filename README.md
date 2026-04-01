# Hand-Gesture-Recognition

Overview
The Hand Gesture Recognition Software is an advanced system that detects and interprets hand gestures using computer vision and machine learning techniques. This software enables gesture-based interactions for applications such as gaming, accessibility tools, and smart home controls.

Features
Real-time Gesture Recognition: Detects and classifies hand gestures in real-time.
Machine Learning-based Model: Utilizes deep learning models for accurate recognition.
Multi-platform Support: Works on Windows, Linux, and macOS.
Customizable Gestures: Allows users to define and train new gestures.
Integration with External Applications: Can be connected with other software via API.
Requirements
Python 3.8+
OpenCV
TensorFlow/Keras
Mediapipe
NumPy
Matplotlib (for visualization)
Installation
Clone the repository:
git clone https://github.com/Saanket-Das/Hand-Gesture-Recognition.git
cd hand-gesture-recognition
Run the software:
python test.py
Usage
Start the application.
Position your hand within the camera’s view.
Perform a predefined gesture.
The software will recognize and display the detected gesture.
Customize gestures by training the model with new hand poses.
Training Custom Gestures
To add new gestures:

Collect gesture images using capture.py.
Label the images and store them in the dataset folder.
Train the model using train.py:
python train.py
API Integration
The software provides an API to integrate with external applications. Example request:

POST /api/recognize
Content-Type: application/json
{
  "image": "base64-encoded-image-data"
}
