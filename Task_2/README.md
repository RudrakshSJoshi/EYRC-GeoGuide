# Task 2A Readme

## Overview

This script is designed to implement Task 2A of the Geo Guide (GG) Theme for the eYRC 2023-24 competition. It detects ArUco markers in an image, extracts their details such as center coordinates and orientation, and marks them for visualization purposes.

## Disclaimer

This software is provided on an "AS IS WHERE IS BASIS". The licensee/end user is responsible for any and all claims arising from the use of this software or any breach of the terms of the agreement.

## Team Information

- **Team ID**: [Team-ID]
- **Authors**: [Names of team members worked on this file separated by Comma: Name1, Name2, ...]

## File Details

- **Filename**: task_2a.py
- **Functions**:
    - `detect_ArUco_details(image)`

## Dependencies

This script relies on the following modules:
- numpy
- cv2 (OpenCV)
- math

## How to Use

1. Ensure you have the necessary dependencies installed.
2. Place your image files in the specified directory.
3. Update the `img_dir_path` variable in the script to point to your image directory.
4. Run the script.
5. The script will process each image in the directory, detect ArUco markers, and display the marked images.

## Functionality

### `detect_ArUco_details(image)`

- **Purpose**: This function detects ArUco markers in an image, retrieves their details, and returns them as dictionaries.
- **Input**: 
    - `image`: A numpy array representing the input image.
- **Returns**:
    - `ArUco_details_dict`: A dictionary containing the details regarding the ArUco markers. Keys are marker IDs, and values are lists containing center coordinates and angles.
    - `ArUco_corners`: A dictionary containing the corner coordinates of the ArUco markers.
- **Example Usage**:
    ```python
    ArUco_details_dict, ArUco_corners = detect_ArUco_details(image)
    ```

### `mark_ArUco_image(image, ArUco_details_dict, ArUco_corners)`

- **Purpose**: This function marks detected ArUco markers on the input image for visualization purposes.
- **Input**:
    - `image`: Input image with detected ArUco markers.
    - `ArUco_details_dict`: Dictionary containing details regarding the ArUco markers.
    - `ArUco_corners`: Dictionary containing corner coordinates of the ArUco markers.
- **Returns**:
    - Modified image with markers visualized.
- **Example Usage**:
    ```python
    img = mark_ArUco_image(img, ArUco_details_dict, ArUco_corners)
    ```

## Example Output

For each input image, the script will print the detected ArUco details and display the marked image.

## Note

- Ensure the image files are in the correct format and located in the specified directory.
- This script assumes that ArUco markers are present in the input images and are detectable.
- Modifications may be required based on specific requirements or variations in input images.



# Task 2B: Event Classification and Image Analysis

This repository contains code for implementing event classification and image analysis as a part of the GeoGuide(GG) Theme in eYRC 2023-24.

## Contents

### Script 1: `task_2b.py`

#### Mandatory Information

- **Team ID**: GG_3047
- **Author List**: Aaditya Porwal, Rudraksh Sachin Joshi
- **Filename**: task_2b.py
- **Functions**: `classify_event(image)`

#### Description

This script loads a trained model and classifies the event from an image.

### Script 2: `model_training.ipynb`

#### Description

This Jupyter Notebook contains code for training a deep learning model for event classification using a provided dataset.

## Instructions

### For `task_2b.py`

1. Clone this repository to your local machine.
2. Run the `task_2b.py` script using Python.
3. The script will classify events from input images and generate an output file.

### For `model_training.ipynb`

1. Open the Jupyter Notebook environment.
2. Execute the cells in the notebook to train the deep learning model.
3. Save the trained model and weights for later use.

## Additional Information

- **Event Names**: The script uses predefined event names for classification. Do not change these event names.


# Task 2C: Event Identification on Arena Image

This script is for implementing Task 2C of the GeoGuide(GG) Theme in eYRC 2023-24.

## Mandatory Information

- **Team ID**: GG_3047
- **Author List**: Aaditya Porwal
- **Filename**: task_2c.py
- **Functions**: `classify_event(image)`

## Contents

### Script: `task_2c.py`

#### Description

This script identifies events on an arena image and classifies them using a trained deep learning model.

## Instructions

1. Clone this repository to your local machine.
2. Ensure all required dependencies are installed.
3. Run the `task_2c.py` script using Python.
4. The script will process the arena image, identify events, classify them, and generate an output file.

## Additional Information

- **Event Names**: The script uses predefined event names for classification. Do not change these event names.



# Task 2D: AR Marker Tracking and Data Logging

This script is for implementing Task 2D of the GeoGuide(GG) Theme in eYRC 2023-24.

## Mandatory Information

- **Team ID**: eYRC#GG#3047
- **Author List**: Sanskriti, Aaditya Porwal, Arnab Mitra, Rudraksh Sachin Joshi
- **Filename**: Task_2D.py
- **Functions**: `read_csv`, `write_csv`, `tracker`, `main`

## Contents

### Script: `Task_2D.py`

#### Description

This script tracks AR markers and logs their coordinates in a CSV file.

## Instructions

1. Clone this repository to your local machine.
2. Ensure all required dependencies are installed.
3. Run the `Task_2D.py` script using Python.
4. The script will read AR marker coordinates from `lat_long.csv`, track them, and write the tracked coordinates to `live_data.csv`.
5. Test cases will be executed to verify the correctness of the tracking and logging.

## Additional Information

- **CSV File Paths**:
    - `lat_long.csv`: Path to the CSV file containing AR marker coordinates.
    - `live_data.csv`: Path to the CSV file where tracked coordinates will be logged.

