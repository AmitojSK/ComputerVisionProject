# Computer Vision Age, Gender, and Emotion Predictor

This project uses deep learning and computer vision to predict a person's age group, gender, and emotion from webcam images. It leverages OpenCV for image processing and Keras for building and training convolutional neural networks.

## Features

- **Age Prediction:** Classifies faces into 12 age groups (0-2, 3-6, ..., 60+).
- **Gender Prediction:** Classifies faces as Male or Female.
- **Emotion Prediction:** Classifies faces as Neutral, Happy, or Sad.
- **Real-time Webcam Demo:** Uses your webcam to detect faces and display predictions live.

## Project Structure

- `Predictor.ipynb`: Main Jupyter notebook containing all code for data loading, preprocessing, model training, and real-time prediction.
- `model-best/`: Directory containing training images for gender and emotion.
- `model-best/Ages/`: Directory containing training images for each age group.
- `model-best/Training/`: Directory containing gender training images (`male`, `female`).
- `model-best/images/images/train/`: Directory containing emotion training images (`neutral`, `happy`, `sad`).

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Keras
- scikit-learn

Install dependencies with:

```sh
pip install numpy opencv-python keras scikit-learn
```

## Usage

1. **Prepare Data:**  
   Organize your images in the folder structure as shown above.

2. **Train Models:**  
   Run the cells in [Predictor.ipynb](Predictor.ipynb) to:
   - Load and preprocess images
   - Train the age, gender, and emotion models
   - Save the best models

3. **Real-time Prediction:**  
   The notebook includes code to start your webcam, detect faces, and display predictions for age, gender, and emotion.

## Model Details

- **CNN Architecture:**  
  Each model uses two Conv2D layers, MaxPooling, Dropout, Flatten, and Dense layers.
- **Input:**  
  Grayscale face images resized to 100x100 pixels.
- **Output:**  
  - Age: 12 classes (softmax)
  - Gender: 2 classes (softmax)
  - Emotion: 3 classes (softmax)

## Example

When running the webcam demo, the application will:
- Draw a rectangle around detected faces.
- Display predicted gender, age group, and emotion above/below the face.

## Notes

- Make sure your training data is properly labeled and organized.
- You may need to adjust the number of epochs or model parameters for better accuracy.
