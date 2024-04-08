# Autonomous Event Detection and Classification System

## Overview
This repository contains the code for an autonomous event detection and classification system developed for the GeoGuide(GG) Theme (eYRC 2023-24) competition. The system utilizes computer vision techniques and machine learning algorithms to detect and classify events occurring within an arena environment.

## Features
- Real-time event detection and classification
- ArUco marker detection for perspective alignment
- Event cropping to isolate regions of interest
- Machine learning-based event classification
- User-friendly interface with live event labels

## Components
- Camera: Captures live feed of the arena environment
- ESP32 Microcontroller: Controls system operation and communication
- ArUco Markers: Used for perspective alignment and boundary detection
- OpenCV: Computer vision library for image processing tasks
- TensorFlow: Machine learning framework for event classification

## Getting Started
1. Clone this repository: `git clone https://github.com/username/repository.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Connect the camera and ESP32 microcontroller to your system.
4. Run the main script: `python main.py`

## Usage
- Ensure that the camera has a clear view of the arena environment.
- Launch the system and observe the live feed for event detection and classification.
- Monitor the displayed event labels for real-time updates on detected events.

## Next Steps
- Implement additional features such as event tracking and logging.
- Optimize the system for improved performance and accuracy.
- Experiment with different machine learning models and algorithms.

## Acknowledgements
We would like to express our gratitude to [Name of Institution/University] for providing us with the opportunity to participate in the GeoGuide(GG) Theme (eYRC 2023-24) competition. Additionally, we acknowledge the support and guidance of our mentors and advisors throughout the development process.

## Disclaimer
This project is developed as part of an educational initiative and may contain experimental features. Use at your own risk.

