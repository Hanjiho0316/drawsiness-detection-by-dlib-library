Drowsiness Detection by Dlib Library

This project implements a drowsiness detection system using the Dlib library in Python.
The system monitors a person’s eyes through a webcam feed and detects signs of drowsiness (such as prolonged eye closure) in real time.

Features

Face and landmark detection using Dlib

Eye aspect ratio (EAR) calculation to measure eye closure

Real-time webcam feed monitoring

Alerts when drowsiness is detected

Requirements

Python 3.x

Dlib

OpenCV

NumPy

Install the required libraries with:

pip install opencv-python dlib numpy

Usage

Clone the repository:

git clone https://github.com/your-username/drawsiness-detection-by-dlib-library.git


Run the script:

python main.py


The system will start your webcam and display the detection window.

Applications

Driver drowsiness monitoring

Workplace safety systems

Real-time fatigue detection

Notes

Ensure you have Dlib properly installed (it may require CMake and Visual Studio Build Tools on Windows).

The accuracy of detection depends on lighting conditions and webcam quality.
