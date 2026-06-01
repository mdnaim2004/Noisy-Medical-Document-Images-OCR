import os
import torch
import torch.nn as nn
from PIL import Image

from flask import Flask, render_template, request
from torchvision import models, transforms


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

device = torch.device("cpu")

class_names = {
    0: "Medical Bill",
    1: "Discharge Summary"
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = models.resnet18(weights=None)

model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(512, 2)
)

model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()

    return class_names[pred]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            prediction = predict_image(image_path)

    return render_template(
        "index.html",
        prediction=prediction,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(debug=True)