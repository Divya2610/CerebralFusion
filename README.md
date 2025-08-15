# Brain Tumor Classification using Hybrid Deep Learning Models

A comprehensive machine learning project that combines multiple deep learning architectures and traditional ML algorithms to classify brain tumors from MRI images with high accuracy.

## 🧠 Project Overview

This project implements a hybrid approach for brain tumor classification using multiple state-of-the-art models including VGG16, ResNet50, DenseNet121, custom CNN, SVM, and Random Forest. The system achieves superior performance through ensemble methods including stacking, weighted averaging, and majority voting.

## 📊 Dataset

The project uses a brain tumor dataset with four classes:
- **Glioma** - Malignant brain tumor
- **Meningioma** - Usually benign brain tumor
- **No Tumor** - Healthy brain scans
- **Pituitary** - Pituitary adenoma

### Dataset Structure
```
Brain_Tumor_Dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
├── Validation/
└── Testing/
```

## 🏗️ Architecture

### Deep Learning Models
1. **VGG16** - Transfer learning with ImageNet weights
2. **ResNet50** - Residual network architecture
3. **DenseNet121** - Dense connectivity pattern
4. **Custom CNN** - Lightweight convolutional network

### Traditional ML Models
1. **SVM** - Support Vector Machine with Gabor feature extraction
2. **Random Forest** - Ensemble method with HOG features

### Hybrid Ensemble Methods
- **Weighted Averaging** - Performance-based weight assignment
- **Stacking** - Meta-learner approach using Logistic Regression
- **Majority Voting** - Democratic prediction combination

## 🔧 Features

- **Advanced Preprocessing**: OTSU thresholding, image augmentation
- **Feature Extraction**: Gabor filters for SVM, HOG features for Random Forest
- **Model Persistence**: Automatic save/load functionality with Google Drive integration
- **Performance Evaluation**: Comprehensive metrics (Accuracy, F1-Score, AUC-ROC)
- **Real-time Prediction**: Upload and classify new brain MRI images

## 📈 Performance Results

### Individual Model Performance
| Model | Accuracy | F1-Score | AUC-ROC |
|-------|----------|----------|---------|
| VGG16 | 23.16% | 8.71% | 76.89% |
| ResNet50 | 24.21% | 10.81% | 48.21% |
| DenseNet121 | 40.23% | 34.03% | 71.26% |
| Custom CNN | 23.16% | 8.71% | 35.04% |
| SVM | 23.16% | 8.71% | 66.39% |
| **Random Forest** | **92.63%** | **92.62%** | **99.22%** |

### Hybrid Model Performance
| Method | Accuracy | F1-Score | AUC-ROC | Test Accuracy |
|--------|----------|----------|---------|---------------|
| Weighted Averaging | 80.23% | 80.68% | 98.76% | - |
| **Stacking** | **92.98%** | **92.99%** | **99.33%** | **87.03%** |
| Majority Voting | 40.82% | 35.66% | 97.55% | - |

**Best Model**: Stacking ensemble achieving **87.03%** test accuracy

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- scikit-learn
- OpenCV
- scikit-image

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification

# Install dependencies
pip install tensorflow scikit-learn opencv-python scikit-image pandas numpy joblib

# For Google Colab users
from google.colab import drive
drive.mount('/content/drive')
```

## 🚀 Usage

### Training the Models
1. **Prepare Dataset**: Upload your brain tumor dataset to Google Drive
2. **Run Training**: Execute the notebook cells sequentially
3. **Model Selection**: The system automatically selects the best performing models
4. **Hybrid Training**: Top 3 models are combined using ensemble methods

### Making Predictions
```python
# Upload and predict on new images
from google.colab import files
uploaded = files.upload()
image_path = next(iter(uploaded))

# Get prediction
prediction = predict_with_hybrid_model(image_path)
print("Predicted Class:", prediction)
```

### Model Architecture Details

#### Deep Learning Pipeline
- **Input Size**: 224×224×3 RGB images
- **Preprocessing**: Normalization, augmentation (rotation, shift, zoom, flip)
- **Transfer Learning**: Pre-trained ImageNet weights with frozen base layers
- **Fine-tuning**: Custom classification heads

#### Feature Extraction Pipeline
- **Gabor Features**: Multi-orientation, multi-scale texture analysis for SVM
- **HOG Features**: Histogram of Oriented Gradients for Random Forest
- **Image Processing**: OTSU thresholding, grayscale conversion

## 📁 File Structure
```
project/
├── Brain_tumor_hybrid.ipynb    # Main notebook
├── models/                     # Saved model files
│   ├── vgg16_model.h5
│   ├── resnet50_model.h5
│   ├── densenet121_model.h5
│   ├── custom_cnn_model.h5
│   ├── svm_model.pkl
│   ├── random_forest_model.pkl
│   └── stacking_meta_learner.pkl
└── README.md
```

## 🔬 Technical Implementation

### Data Augmentation
- Rotation: ±20 degrees
- Width/Height shift: 0.2
- Shear transformation: 0.2
- Zoom: 0.2
- Horizontal flip: Enabled

### Ensemble Strategy
The stacking approach uses:
1. **Base Models**: Top 3 performing models (Random Forest, DenseNet, VGG16)
2. **Meta-learner**: Logistic Regression
3. **Cross-validation**: Robust validation strategy
4. **Feature Stacking**: Concatenation of prediction probabilities

## 📊 Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Weighted average F1-score for multi-class
- **AUC-ROC**: Area under ROC curve (multi-class OvR)
- **Combined Score**: Average of normalized metrics

## 🎯 Key Achievements
- ✅ **87.03%** test accuracy with stacking ensemble
- ✅ **Multi-modal approach** combining deep learning and traditional ML
- ✅ **Robust feature extraction** using Gabor and HOG descriptors
- ✅ **Automated model selection** based on performance metrics
- ✅ **Real-time prediction** capability
- ✅ **Model persistence** with Google Drive integration

## 🔮 Future Enhancements
- [ ] Implement attention mechanisms
- [ ] Add LIME/SHAP explainability
- [ ] Develop web-based deployment
- [ ] Include more tumor types
- [ ] 3D MRI volume analysis
- [ ] Integration with medical imaging standards (DICOM)

## 📚 Dependencies
```
tensorflow>=2.8.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
scikit-image>=0.19.0
pandas>=1.3.0
numpy>=1.21.0
joblib>=1.1.0
scipy>=1.7.0
matplotlib>=3.5.0
```

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- Transfer learning models from TensorFlow/Keras
- Brain tumor dataset contributors
- Google Colab for computational resources
- Open source ML community


⚠️ **Medical Disclaimer**: This tool is for research and educational purposes only. It should not be used for actual medical diagnosis without proper validation and approval from medical professionals.
