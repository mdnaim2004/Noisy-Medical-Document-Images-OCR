# Noisy Medical Document Images OCR

This project classifies noisy medical document images into two labels:

- Medical Bill
- Discharge Summary

The main notebook walks through the full training pipeline, and the Flask app in `medical_doc_app/` serves the trained model for local image uploads.

## Project Flow

1. Load and inspect the dataset in the notebook.
2. Preprocess and augment the images.
3. Train a ResNet18-based classifier.
4. Evaluate the model with validation metrics and a confusion matrix.
5. Save the final weights as `model.pth`.
6. Move or copy `model.pth` into `medical_doc_app/`.
7. Run the Flask app and upload a document image for prediction.

## Notebook

Open [noisy-medical-document-images-ocr.ipynb](noisy-medical-document-images-ocr.ipynb) to see the full end-to-end workflow, from dataset handling to model export and deployment notes.

## Flask App

The web app lives in [medical_doc_app/app.py](medical_doc_app/app.py). It loads `medical_doc_app/model.pth`, preprocesses uploaded images, and returns the predicted class.

## Run the App

From inside the `medical_doc_app/` folder:

```bash
python app.py
```

Then open the local address shown in the terminal, usually:

```text
http://127.0.0.1:5000/
```

Upload a medical document image and the app will display the predicted category.

## Kaggle Reference

Original notebook reference:

[Noisy Medical Document Images (OCR)](https://www.kaggle.com/code/mdnaimislam165436/noisy-medical-document-images-ocr)
