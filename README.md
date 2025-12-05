# MNIST Handwritten Digit Classification (CNN)

Context
-------
This project demonstrates a convolutional neural network (CNN) for classifying handwritten digits (0–9) using the MNIST dataset. It is intended as an instructional implementation and a lightweight reference for training, evaluating, and exporting a small CNN suitable for production use (TensorFlow Lite).

Key goals
- Load and preprocess the MNIST dataset.
- Build and train a compact, high-accuracy CNN.
- Evaluate model performance (accuracy, loss, confusion matrix, learning curves).
- Export the trained model to TensorFlow Lite for efficient deployment.

Dataset
-------
- MNIST (70,000 grayscale images, 28×28 px)
- Training: 60,000 images
- Test: 10,000 images

Prerequisites & Technologies
----------------------------
- OS: Linux (development tested)
- Python: 3.7+
- Recommended: virtual environment (venv)

Required Python packages
- tensorflow (includes Keras API)
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter (optional, for the notebook)

Install (example)
```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install tensorflow numpy matplotlib seaborn scikit-learn jupyter
```

Project artifacts
-----------------
- Jupyter notebook: `classificationChiffreMeilleurCode.ipynb` — full workflow: load, preprocess, train, evaluate, export.
- Export folder: `export/mnist_cnn.tflite` — TensorFlow Lite model for fast, small-footprint inference.

Quick start
-----------
1. Activate environment.
2. Launch the notebook:
   jupyter notebook classificationChiffreMeilleurCode.ipynb
3. Run cells to train and export the model. The notebook includes commands to save a quantized TFLite model.

Notes
-----
- Images are normalized to [0, 1] and reshaped to (28, 28, 1).
- Labels are one-hot encoded for categorical cross-entropy training.
- The exported TFLite model is quantized by default (smaller size, minimal accuracy loss on MNIST).

Author
------
Bonaventure DANFI — 2025
