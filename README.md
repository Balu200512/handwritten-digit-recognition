# Handwritten Digit Recognition Using CNN

## 1. Project Overview

Handwritten Digit Recognition is a machine learning project that identifies handwritten numerical digits from 0 to 9 using a Convolutional Neural Network (CNN).

The project uses the MNIST dataset, which contains grayscale images of handwritten digits. Images are preprocessed and passed to a trained CNN model to predict the corresponding digit.

The main objective is to demonstrate how deep learning and image classification can be used to recognize handwritten numbers automatically.

## 2. Objectives

- Build a CNN model for handwritten digit classification.
- Preprocess and normalize image data.
- Train a deep learning model using the MNIST dataset.
- Evaluate the model using test data.
- Predict digits from handwritten images.
- Visualize predicted and actual digit labels.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow/Keras | Building and training the CNN |
| CNN | Extracting image features and classifying digits |
| NumPy | Numerical operations and array processing |
| Matplotlib | Visualizing images and predictions |
| MNIST | Handwritten digit dataset |

## 4. Dataset Description

The project uses the MNIST handwritten digit dataset.

- Training images: 60,000
- Testing images: 10,000
- Image dimensions: 28 × 28 pixels
- Image format: Grayscale
- Number of classes: 10 (digits 0–9)

The dataset is loaded through TensorFlow/Keras. Manual downloading is not required when the program has internet access on its first run.

## 5. Project Workflow

1. Load the MNIST dataset.
2. Normalize pixel values to the range 0–1.
3. Reshape images to include the grayscale channel dimension.
4. Build the CNN architecture.
5. Compile the model using the Adam optimizer and sparse categorical cross-entropy loss.
6. Train the model using training images and labels.
7. Evaluate its performance using test images.
8. Predict digits and display sample predictions.
9. Save the trained model for future use.

## 6. CNN Architecture

The model uses the following layers:

- **Input layer:** Accepts 28 × 28 grayscale images.
- **Convolutional layer:** Extracts visual features from digit images.
- **Max-pooling layer:** Reduces spatial dimensions.
- **Second convolutional layer:** Learns additional image features.
- **Second max-pooling layer:** Further reduces feature-map dimensions.
- **Flatten layer:** Converts feature maps into a one-dimensional vector.
- **Dense layer:** Learns patterns useful for classification.
- **Dropout layer:** Helps reduce overfitting during training.
- **Output layer:** Produces probabilities for the ten digit classes.

## 7. Features

- Automatic dataset loading
- Image normalization and preprocessing
- CNN-based digit classification
- Model training and evaluation
- Prediction of handwritten digits
- Visualization of actual versus predicted labels
- Saving the trained model in Keras format

## 8. Installation Requirements

Install Python and the required libraries.

```bash
pip install -r requirements.txt
```

## 9. How to Run the Project

Clone the repository:

```bash
git clone https://github.com/Balu200512/handwritten-digit-recognition.git
```

Open the project directory:

```bash
cd handwritten-digit-recognition
```

Run the training script:

```bash
python train.py
```

The script downloads the MNIST dataset if necessary, trains the CNN, evaluates it, displays sample predictions, and saves the trained model as `digit_cnn.keras`.

## 10. Expected Output

When the program runs successfully, it:

- Displays the CNN model training progress.
- Prints the test loss and test accuracy.
- Shows sample handwritten digit images with predicted and actual labels.
- Saves the trained model as `digit_cnn.keras`.

Actual accuracy and execution time depend on the training environment and model configuration.

## 11. Applications

- Digit recognition in scanned documents
- Reading handwritten numerical entries
- Educational demonstrations of deep learning
- Introduction to image classification
- Optical character recognition research prototypes

## 12. Limitations

- The model is trained on the MNIST dataset and may not perform equally well on all real-world handwriting.
- It is designed for isolated digits rather than complete handwritten sentences.
- Performance depends on image quality and preprocessing.
- The model's accuracy must be measured by running the training and evaluation code.

## 13. Future Enhancements

- Build a graphical interface for drawing digits.
- Allow users to upload their own digit images.
- Add real-time digit recognition.
- Experiment with CNN architectures and hyperparameters.
- Deploy the model as a web application.

## 14. Author

**Balu200512**

GitHub: https://github.com/Balu200512

## 15. Conclusion

This project demonstrates the use of Convolutional Neural Networks for handwritten digit recognition using Python and TensorFlow/Keras. It covers dataset preparation, image preprocessing, model training, evaluation, prediction, and visualization in a practical machine learning workflow.
