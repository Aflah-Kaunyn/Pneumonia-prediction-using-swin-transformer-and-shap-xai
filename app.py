from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import timm
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import cv2
import os

app = Flask(__name__)

# ------------------ MODEL DEFINITION ------------------ #
class SwinTransformerPneumonia(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(SwinTransformerPneumonia, self).__init__()
        self.model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)

# ------------------ LOAD MODEL ------------------ #
model = SwinTransformerPneumonia()
model.load_state_dict(torch.load("swin_pneumonia.pth", map_location=torch.device("cpu")))
model.eval()

# ------------------ IMAGE TRANSFORM ------------------ #
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

LABELS = ["Normal", "Bacterial Pneumonia", "Viral Pneumonia"]

# ------------------ HELPER FUNCTIONS ------------------ #
def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def denormalize_image(tensor):
    img_np = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np * std + mean
    return np.clip(img_np, 0, 1)

def create_shap_visualization(input_tensor, model, predicted_class):
    try:
        background_samples = [input_tensor]
        noise = torch.randn_like(input_tensor) * 0.05
        background_samples.append(torch.clamp(input_tensor + noise, -3, 3))
        background = torch.cat(background_samples, dim=0)

        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(input_tensor, nsamples=30)

        shap_arr = np.array(shap_values)
        if len(shap_arr.shape) == 5:
            shap_for_class = shap_arr[0, predicted_class, :, :, :]
        elif len(shap_arr.shape) == 4:
            shap_for_class = shap_arr[predicted_class if shap_arr.shape[0] == 3 else 0, :, :, :]
        else:
            shap_for_class = shap_arr[0]

        shap_importance = np.mean(np.abs(shap_for_class), axis=2) if len(shap_for_class.shape) == 3 else np.abs(shap_for_class)
        img_np = denormalize_image(input_tensor)
        img_gray = np.mean(img_np, axis=2)

        p_low, p_high = np.percentile(shap_importance, 1), np.percentile(shap_importance, 99)
        shap_normalized = np.clip((shap_importance - p_low) / (p_high - p_low + 1e-8), 0, 1)
        shap_normalized = cv2.GaussianBlur(shap_normalized, (3, 3), 0.5)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_gray, cmap='gray')
        axes[0].set_title('Original X-ray', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        im = axes[1].imshow(shap_importance, cmap='RdBu_r',
                            vmin=np.percentile(shap_importance, 10),
                            vmax=np.percentile(shap_importance, 90))
        axes[1].set_title('Feature Importance Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        overlay = np.stack([img_gray]*3, axis=2) * 0.92 + np.expand_dims(shap_normalized, 2) * np.array([1,0,0]) * 0.08
        axes[2].imshow(np.clip(overlay,0,1))
        axes[2].set_title('SHAP Overlay', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor='white')
        buf.seek(0)
        comparison = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()
        return comparison

    except Exception as e:
        print(f"SHAP error: {e}")
        return None

# ------------------ ROUTES ------------------ #
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or request.files["file"].filename == "":
        return render_template("index.html", error="No file selected")

    file = request.files["file"]
    save_path = os.path.join("static", file.filename)
    file.save(save_path)

    image = Image.open(save_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence_score, predicted_class = torch.max(probabilities, 1)
        prediction = LABELS[predicted_class.item()]
        confidence = f"{confidence_score.item() * 100:.2f}%"
        all_probabilities = {
            label: f"{prob * 100:.2f}%" 
            for label, prob in zip(LABELS, probabilities[0].tolist())
        }

    #  prediction
    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           all_probabilities=all_probabilities,
                           filename=file.filename,
                           labels=LABELS)

@app.route("/explain", methods=["POST"])
def explain():
    filename = request.form["filename"]
    predicted_label = request.form["prediction"]
    predicted_class = LABELS.index(predicted_label)

    image_path = os.path.join("static", filename)
    input_tensor = preprocess(image_path)
    shap_visualization = create_shap_visualization(input_tensor, model, predicted_class)

    return render_template("index.html",
                           prediction=predicted_label,
                           shap_visualization=shap_visualization)

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True, use_reloader=False)
