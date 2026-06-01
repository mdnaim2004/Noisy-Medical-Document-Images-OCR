# Noisy Medical Document Images OCR

A deep learning project for classifying noisy medical document images using OCR techniques. Built with PyTorch and Flask, this project provides both a Jupyter notebook for model training and a web application for inference.

## 📋 Overview

This project tackles the challenge of classifying degraded or noisy medical document images into two categories:

- **Medical Bill** - Hospital/healthcare facility billing documents
- **Discharge Summary** - Patient discharge summaries and clinical notes

The trained ResNet18-based classifier achieves robust performance on noisy, scanned medical documents, making it suitable for document digitization pipelines.

## 🗂️ Project Structure

```
.
├── README.md                              # This file
├── noisy-medical-document-images-ocr.ipynb # Training notebook
└── medical_doc_app/
    ├── app.py                             # Flask web application
    ├── model.pth                          # Trained model weights (ResNet18)
    ├── templates/                         # HTML templates for web UI
    └── uploads/                           # Temporary directory for uploaded images
```

## 🔄 Project Flow

The project follows a complete ML pipeline from training to deployment:

1. **Data Preparation**: Load and inspect the medical document dataset
2. **Preprocessing**: Resize, normalize, and augment images for robust training
3. **Model Training**: Train a ResNet18-based classifier on noisy documents
4. **Evaluation**: Assess model performance with validation metrics and confusion matrix
5. **Export**: Save trained weights as `model.pth`
6. **Deployment**: Serve the model via a Flask web application

## 📓 Training Notebook

The main notebook ([noisy-medical-document-images-ocr.ipynb](noisy-medical-document-images-ocr.ipynb)) contains the full end-to-end workflow:

- Dataset loading and exploration
- Image preprocessing and augmentation strategies
- Model architecture and training configuration
- Validation metrics and confusion matrix visualization
- Model export and deployment guidance

## 🚀 Running the Web Application

### Prerequisites

- Python 3.7+
- Flask
- PyTorch
- torchvision
- Pillow

### Installation

```bash
pip install flask torch torchvision pillow
```

### Start the Application

Navigate to the `medical_doc_app/` directory and run:

```bash
python app.py
```

The application will start on:

```
http://127.0.0.1:5000/
```

Open this URL in your browser to access the web interface.

### Using the Application

1. Click the upload button or drag-and-drop a medical document image
2. The application preprocesses the image automatically
3. The trained model generates a prediction (Medical Bill or Discharge Summary)
4. Results are displayed on the web interface

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.gif`

## 🔧 Flask App Details

**File**: [medical_doc_app/app.py](medical_doc_app/app.py)

The Flask application:
- Loads the pre-trained ResNet18 model (`model.pth`)
- Handles image uploads through a simple web interface
- Preprocesses images to match training specifications
- Returns predictions with confidence scores
- Stores temporary uploads in the `uploads/` folder

## 📊 Model Architecture

- **Base Model**: ResNet18 (pre-trained on ImageNet)
- **Output Classes**: 2 (Medical Bill, Discharge Summary)
- **Input Size**: Images resized to 224×224 pixels
- **Training Framework**: PyTorch

## 🎯 Use Cases

- **Document Digitization**: Automatically classify scanned medical records
- **Medical Workflow Automation**: Route documents to appropriate processing pipelines
- **Document Management**: Sort large batches of medical documents efficiently
- **Healthcare IT**: Integrate into EMR/EHR systems for document classification

## 📚 References

**Original Kaggle Notebook:**  
[Noisy Medical Document Images (OCR)](https://www.kaggle.com/code/mdnaimislam165436/noisy-medical-document-images-ocr)

## 💡 Tips for Best Results

- Use clear, well-lit scans or photos of medical documents
- For very noisy or severely degraded images, consider additional preprocessing
- The model performs best with standard document orientations
- Batch processing can be implemented for high-throughput scenarios

## 📝 License

Please refer to the original Kaggle dataset for licensing information.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page for ways to contribute.
