# Air Quality Prediction

## Overview
This project is a machine learning model built using TensorFlow and Keras to predict air quality based on various environmental features. The model is trained on a dataset containing multiple features, normalizes the input data, and uses a neural network to classify air quality into different categories.

## Features
- Data preprocessing with feature scaling using `MinMaxScaler`.
- Neural network model with multiple dense layers, batch normalization, dropout, and L2 regularization.
- Training with early stopping and learning rate reduction techniques.
- Categorical classification of air quality.
- Evaluation of the model's accuracy on test data.
- Predicting air quality for new input data.

## Dataset
The dataset is expected to be in a CSV file named `dataset.csv`, with the following columns:
```
Feature1, Feature2, Feature3, Feature4, Feature5, Feature6, AirQuality
```
- `Feature1` to `Feature6` represent environmental parameters.
- `AirQuality` is the target variable with categorical values.

## Installation & Dependencies
To run this project, install the required dependencies using:
```bash
pip install pandas scikit-learn tensorflow
```

## How to Run
1. Ensure that `dataset.csv` is available in the working directory.
2. Run the Python script:
   ```bash
   python air_quality_dataset.py
   ```
3. The script will:
   - Load and preprocess the dataset.
   - Train a neural network model.
   - Evaluate the model on test data.
   - Predict air quality for a sample input.

## Model Architecture
The model consists of:
- Input layer with 256 neurons and ReLU activation.
- Hidden layers with 128 and 64 neurons.
- Dropout and Batch Normalization for regularization.
- Output layer with softmax activation for classification.

## Prediction Example
A sample prediction is made using:
```python
new_data = [[640, 590, 1105, 1608, 1459, 2427]]
new_data_normalized = scaler.transform(new_data)
prediction = model.predict(new_data_normalized)
predicted_class = prediction.argmax(axis=1) + 1
print(f"Predicted Air Quality Class: {predicted_class}")
```

## License
This project is open-source and available under the MIT License.

## Author
[Dilyan Grigorov]

## Acknowledgments
- TensorFlow and Keras for deep learning.
- Scikit-learn for preprocessing and model evaluation.
- Pandas for data manipulation.


