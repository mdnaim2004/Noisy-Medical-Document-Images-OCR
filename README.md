# Noisy Medical Document Images OCR

This project classifies noisy medical document images into two document types:

- Medical Bill
- Discharge Summary

It contains two parts:

- A Jupyter notebook for the full training and evaluation pipeline
- A Flask web application for local inference on uploaded images

The project is based on a ResNet18 deep learning model trained to recognize medical document categories even when the images are noisy, blurred, scanned poorly, or affected by shadows and artifacts.

## Project Goal

The main goal is to build an end-to-end pipeline that can:

1. Load and inspect the medical document dataset
2. Preprocess and augment the images
3. Train a classification model
4. Evaluate the model on validation data
5. Save the trained weights
6. Run a Flask app that predicts the document class from an uploaded image

## What This Project Solves

Medical documents often come from real-world scans or phone photos. Those images are not clean. They can contain:

- Blur
- Noise
- Shadows
- Uneven lighting
- Skewed orientation
- Low contrast

This project is designed to handle those conditions and still classify the document into the correct category.

## Repository Structure

```
.
├── README.md
├── noisy-medical-document-images-ocr.ipynb
└── medical_doc_app/
    ├── app.py
    ├── model.pth
    ├── templates/
    │   └── index.html
    └── uploads/
```

### File Purpose

- [noisy-medical-document-images-ocr.ipynb](noisy-medical-document-images-ocr.ipynb): training, evaluation, and model export notebook
- [medical_doc_app/app.py](medical_doc_app/app.py): Flask inference app
- [medical_doc_app/model.pth](medical_doc_app/model.pth): saved trained model weights
- [medical_doc_app/templates/index.html](medical_doc_app/templates/index.html): web UI template
- [medical_doc_app/uploads/](medical_doc_app/uploads/): temporary folder for uploaded files

## End-to-End Workflow

The complete pipeline follows this order:

### 1. Dataset Loading

The notebook starts by loading the dataset and checking the folder structure. This ensures the class folders and image paths are available before training begins.

### 2. Preprocessing

Each image is resized to 224 × 224 and normalized using ImageNet statistics. This matches the input format expected by ResNet18.

### 3. Model Building

The notebook uses a ResNet18 backbone with a custom classification head for two output classes.

### 4. Training

The model is trained on the medical document images using PyTorch.

### 5. Validation

The notebook evaluates model performance using metrics such as accuracy, precision, recall, F1-score, and a confusion matrix.

### 6. Export

After training, the final weights are saved as `model.pth`.

### 7. Deployment

The Flask app loads `model.pth`, accepts an uploaded image, preprocesses it, and returns a prediction.

## Notebook Walkthrough

Open [noisy-medical-document-images-ocr.ipynb](noisy-medical-document-images-ocr.ipynb) to see the full workflow.

The notebook includes:

- Environment setup
- Dataset path checking
- Dataset exploration
- Image visualization
- Label distribution analysis
- Train/validation split
- Transform and augmentation setup
- PyTorch dataset and dataloader creation
- ResNet18 model definition
- Training loop
- Validation and metric calculation
- Confusion matrix plotting
- Saving the final model

## How the Flask App Works

The app in [medical_doc_app/app.py](medical_doc_app/app.py) performs these steps:

1. Accepts an uploaded image from the browser
2. Saves the file temporarily inside the `uploads/` folder
3. Opens the image and converts it to RGB
4. Applies the same preprocessing used during training
5. Loads the trained ResNet18 weights from `model.pth`
6. Runs inference on the image
7. Displays the predicted class

### Supported Classes

- Medical Bill
- Discharge Summary

## How to Run the App

### Prerequisites

Make sure these are installed:

- Python 3.11 or compatible version
- PyTorch
- torchvision
- Flask
- Pillow

### Install Dependencies

If needed, install the required packages:

```bash
pip install flask torch torchvision pillow
```

### Run the App

Go to the `medical_doc_app/` folder and start the Flask server:

```bash
python app.py
```

The app will usually run at:

```text
http://127.0.0.1:5000/
```

Open that address in your browser.

### Important Run Note

The app expects `model.pth` to be available in the same folder as `app.py`. If you retrain the model in the notebook, copy the new weights into `medical_doc_app/` before running the app.

## How to Use the App

1. Open the local web page in your browser.
2. Upload a medical document image.
3. Wait for the model to process the image.
4. Read the predicted class shown on the page.

The app is meant for single-image inference and local demonstration.

## Model Details

- Base architecture: ResNet18
- Output classes: 2
- Input size: 224 × 224
- Framework: PyTorch
- Inference device: CPU

The classification head uses a dropout layer followed by a linear layer for binary classification.

## Dataset and Training Notes

The original Kaggle notebook is based on noisy medical document images. In this project, the training setup is designed to improve robustness by using image preprocessing and augmentation.

This helps the model perform better on real document scans where image quality is not ideal.

## Best Practices

- Use clear image uploads when possible
- Avoid heavily cropped document images
- Keep the document orientation upright
- If the model file changes, restart the Flask app
- Run the app from inside `medical_doc_app/` so the relative paths work correctly

## Troubleshooting

### The app does not start

Check whether the required Python packages are installed and whether `model.pth` exists in `medical_doc_app/`.

### The model file cannot be found

Make sure the trained weights are saved at:

`medical_doc_app/model.pth`

### Uploaded image is not predicted correctly

Try using a clearer image, a better scan, or a correctly oriented document photo.

### Browser shows a server error

Check the terminal output where `python app.py` was started. Most Flask errors appear there first.

## Reference

Original Kaggle notebook:

[Noisy Medical Document Images (OCR)](https://www.kaggle.com/code/mdnaimislam165436/noisy-medical-document-images-ocr)

## License

Please check the dataset and original Kaggle notebook licensing terms before redistribution or commercial use.
