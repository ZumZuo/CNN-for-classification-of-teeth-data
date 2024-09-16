from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import io
import tifffile
import base64
import cv2

app = Flask(__name__)

from segmentation_models_pytorch import Unet
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.model = Unet(
            encoder_name='resnet34', 
            encoder_weights='imagenet', 
            in_channels=12,
            classes=1,
        )
        
        self.batch_norm = nn.BatchNorm2d(12)

    def forward(self, x):
        x = self.batch_norm(x)
        output = self.model(x)
        return torch.sigmoid(output)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
model.load_state_dict(torch.load('week4_task.pth', map_location=device))
model.eval()

def preprocess_image(image):
    image = torch.tensor(image).float().permute(2, 0, 1)
    transform = transforms.Compose([
        transforms.Resize((128,128)),
    ])
    return transform(image).unsqueeze(0).to(device)

def array_to_base64(mask, rgb_image):
    mask = (mask * 255).astype(np.uint8)
    overlay = rgb_image.copy()
    overlay[mask > 0] = [0, 0, 255]
    overlay = overlay.astype(np.uint8)
    
    pil_img = Image.fromarray(overlay, 'RGB')
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def upscale_image(image, target_size):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    image = tifffile.imread(file)
    
    rgb_image = image[:, :, 1:4]
    rgb_image = rgb_image[:, :, ::-1]
    rgb_image = rgb_image / (np.max(rgb_image) - np.min(rgb_image))
    rgb_image = (rgb_image*255).astype(np.uint8)

    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)

    predicted_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    rgb_image_upscaled = upscale_image(rgb_image, (256, 256))
    predicted_mask_upscaled = upscale_image(predicted_mask, (256, 256))

    predicted_mask_base64 = array_to_base64(predicted_mask_upscaled, rgb_image_upscaled)
    
    rgb_pil_img = Image.fromarray(rgb_image_upscaled, 'RGB')
    buffer = io.BytesIO()
    rgb_pil_img.save(buffer, format="PNG")
    rgb_image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({'rgb_image': rgb_image_base64, 'mask_overlay': predicted_mask_base64})

if __name__ == '__main__':
    app.run()