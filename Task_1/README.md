# Task 1: Salary Prediction Model and Georeferencing

This repository contains code for implementing a salary prediction model as a part of the GeoGuide(GG) Theme in eYRC 2023-24. Additionally, it includes a georeferenced image added using QGIS.

## Contents

- `task_1a.py`: Python script implementing the model training pipeline.
- `task_1a_dataset.csv`: Dataset containing employee information.
- `task_1b.png`: Georeferenced image added using QGIS.

## Instructions

To run the salary prediction model:

1. Ensure you have Python installed on your system.
2. Clone this repository to your local machine.
3. Navigate to the repository directory.
4. Install the required Python dependencies using `pip install -r requirements.txt`.
5. Run the `task_1a.py` script using Python.
6. The script will preprocess the dataset, train the model, and output the accuracy on the test set.

## Overview of `task_1a.py`

- **Data Preprocessing**: The script preprocesses the dataset by encoding categorical features and standardizing numerical features.
- **Model Definition**: The salary prediction model architecture is defined using PyTorch.
- **Training**: The model is trained using the provided dataset with a custom loss function and Adam optimizer.
- **Validation**: Model accuracy is evaluated on the test set to assess its performance.

## Task 1B: Georeferencing

The `task_1b.png` file contains a georeferenced image added using QGIS. This image can be used for spatial analysis and visualization within GIS software.

## Additional Information

- **Team ID**: eYRC#GG#3047
- **Authors**: Aaditya Porwal, Rudraksh Sachin Joshi
- For more details on the implementation and methodology, please refer to the comments within the `task_1a.py` script.
- Feel free to reach out to the authors for any queries or clarifications.

