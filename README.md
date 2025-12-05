# Classification_Chiffres_MNIST

Handwritten digit classification using convolutional neural networks (CNN) on the MNIST dataset.

## 📋 Context

This project implements a convolutional neural network for automatic classification of handwritten digits (0-9). It uses the MNIST dataset, a standard benchmark containing 70,000 grayscale images of handwritten digits (28×28 pixels).

### Objectives
- Load and preprocess MNIST data
- Build and train a high-performance CNN model
- Evaluate model performance with various metrics
- Export the model for production deployment

## 🔧 Prerequisites & Technologies

### Language
- **Python** 3.7+

### Required Libraries
- **TensorFlow/Keras** - Deep learning framework for building and training CNN models
- **NumPy** - Array manipulation and numerical computations
- **Matplotlib** - Visualization of images and learning curves
- **Seaborn** - Advanced data visualization (confusion matrices)
- **Scikit-learn** - Machine learning metrics (confusion_matrix)

## 📦 Installation

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install tensorflow keras numpy matplotlib seaborn scikit-learn
```

## 🏗️ Model Architecture

The CNN consists of:
- **3 Convolutional layers** (32, 64, 64 filters) with ReLU activation
- **2 MaxPooling layers** (2×2 kernel)
- **Dropout layer** (0.5) to prevent overfitting
- **Fully Connected layer** (64 neurons)
- **Output layer** (10 neurons, softmax activation for multi-class classification)

### Training Configuration
- **Optimizer**: Adam
- **Loss function**: Categorical Cross-Entropy
- **Epochs**: 10
- **Validation split**: 20%

## 📊 Results

After 10 epochs of training:
- **Test Accuracy**: 99.29%
- **Test Loss**: 0.0210
- **Training Accuracy**: 99.23%
- **Validation Accuracy**: 99.31%

## 📁 Project Structure

```
Classification_Chiffres_MNIST/
├── classificationChiffreMeilleurCode.ipynb  # Complete Jupyter notebook
├── README.md                                 # This file
└── export/
    └── mnist_cnn.tflite                     # Exported TensorFlow Lite model
```

## 🚀 Usage

Run the Jupyter notebook:
```bash
jupyter notebook classificationChiffreMeilleurCode.ipynb
```

The model is automatically exported in TensorFlow Lite format (~80-100 KB) for lightweight production deployment.

## 📝 Data Preprocessing

- **Normalization**: Pixel values divided by 255.0 (range 0-1)
- **Reshape**: (60000, 28, 28) → (60000, 28, 28, 1)
- **Label Encoding**: One-hot encoding for digits 0-9

## 📈 Evaluation Metrics

- **Accuracy**: Classification accuracy on test set
- **Loss**: Categorical Cross-Entropy loss
- **Confusion Matrix**: Detailed per-class performance analysis
- **Learning Curves**: Training vs. Validation metrics over epochs

## 🔍 Model Export

The trained model is exported to TensorFlow Lite format for efficient deployment:
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```

Benefits:
- File size: ~80-100 KB (4× smaller than original)
- Minimal accuracy loss after 8-bit quantization
- Ultra-fast inference for Flask or mobile applications

## 📚 References

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Keras Documentation](https://keras.io/)
- [TensorFlow Lite](https://www.tensorflow.org/lite)

---

**Author**: Bonaventure DANFI  
**Created**: 2025
