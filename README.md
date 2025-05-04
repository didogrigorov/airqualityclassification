# Air Quality Prediction

## Overview
This project is a machine learning-based air quality prediction system developed using TensorFlow and Keras. It utilizes a dataset containing various environmental features to classify air quality into different categories. The model is designed for robust performance, employing feature scaling, dropout regularization, and batch normalization to improve accuracy and prevent overfitting.

## Features
- **Data Preprocessing:** Feature scaling using `MinMaxScaler` for normalization.
- **Neural Network Architecture:** Multi-layer perceptron (MLP) with dense layers, batch normalization, dropout, and L2 regularization.
- **Optimization Techniques:** Uses early stopping and learning rate reduction strategies to enhance model performance.
- **Classification:** Predicts air quality levels based on input environmental parameters.
- **Evaluation Metrics:** Reports loss and accuracy on test data.
- **Predictive Capabilities:** Accepts new input data for air quality classification.

## Dataset
The dataset is expected to be in a CSV file named `dataset.csv` and should have the following columns:
```
Feature1, Feature2, Feature3, Feature4, Feature5, Feature6, AirQuality
```
- `Feature1` to `Feature6`: Numerical values representing various environmental factors affecting air quality.
- `AirQuality`: The target variable, categorized into different air quality levels.

## Installation & Dependencies
Ensure you have the required dependencies installed:
```bash
pip install pandas scikit-learn tensorflow
```

## How to Run the Project
1. Place `dataset.csv` in the working directory.
2. Execute the Python script:
   ```bash
   python air_quality_dataset.py
   ```
3. The script performs the following tasks:
   - Loads and preprocesses the dataset.
   - Splits data into training and testing sets.
   - Builds and trains a deep learning model.
   - Evaluates the model's accuracy on test data.
   - Predicts air quality for a sample input.

## Model Architecture
The neural network comprises the following layers:
1. **Input Layer:**
   - 256 neurons with ReLU activation
   - L2 regularization to reduce overfitting
2. **Hidden Layers:**
   - 128 and 64 neurons with ReLU activation
   - Batch normalization to stabilize training
   - Dropout layers (30-40%) to prevent overfitting
3. **Output Layer:**
   - Softmax activation for multi-class classification

### Model Schema
Below is a simple schema representation of the model architecture:
```
Input Layer (256 neurons, ReLU, L2 Regularization)
        |
Batch Normalization
        |
Dropout (40%)
        |
Hidden Layer (128 neurons, ReLU, L2 Regularization)
        |
Batch Normalization
        |
Dropout (30%)
        |
Hidden Layer (64 neurons, ReLU, L2 Regularization)
        |
Batch Normalization
        |
Dropout (30%)
        |
Output Layer (Softmax Activation)
```

## Training Details
- The model is compiled using the **Adam optimizer** with a learning rate of `0.001`.
- Uses **categorical cross-entropy** as the loss function.
- Implements **early stopping** and **ReduceLROnPlateau** to optimize training efficiency.
- Training is conducted for **up to 200 epochs**, with a batch size of **32**.

## Model Evaluation
- The script evaluates model accuracy and loss on the test dataset.
- Accuracy is printed after model evaluation:
  ```python
  loss, accuracy = model.evaluate(X_test, y_test)
  print(f"Test Accuracy: {accuracy}")
  ```

## Making Predictions
A sample prediction can be made as follows:
```python
new_data = [[640, 590, 1105, 1608, 1459, 2427]]
new_data_normalized = scaler.transform(new_data)
prediction = model.predict(new_data_normalized)
predicted_class = prediction.argmax(axis=1) + 1
print(f"Predicted Air Quality Class: {predicted_class}")
```

## Future Enhancements
- **Expand Dataset:** Incorporate more features and data points to improve accuracy.
- **Hyperparameter Tuning:** Experiment with different architectures and optimizers.
- **Deploy Model:** Convert into an API or web-based interface for real-time predictions.
- **Visualization:** Integrate Matplotlib/Seaborn for data analysis and insights.

## License
This project is open-source and available under the MIT License.

## Author
Dilyan Grigorov

## Acknowledgments
- **TensorFlow & Keras:** For deep learning framework.
- **Scikit-learn:** For preprocessing and model evaluation.
- **Pandas:** For data manipulation and processing.
