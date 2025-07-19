# app.py

import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import re
from flask import Flask, request, jsonify, render_template
from model.model import CNN

import base64
import io
from PIL import Image, ImageOps

app = Flask(__name__)

model = CNN()
checkpoint = torch.load('project_new/saved_models/best_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    img_str = re.search(r'base64,(.*)', data).group(1)
    image = Image.open(io.BytesIO(base64.b64decode(img_str))).convert('L')  # 转灰度

    # 反转颜色，白底黑字变成黑底白字（MNIST格式）
    image = ImageOps.invert(image)

    # 调整图像大小为28x28，保证和MNIST一致
    image = image.resize((28, 28))

    # 预处理：归一化、转tensor等
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1).item()

    return jsonify({'prediction': pred})





# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json['image']
#     img_str = re.search(r'base64,(.*)', data).group(1)
#     image = Image.open(io.BytesIO(base64.b64decode(img_str)))
#     image = transform(image).unsqueeze(0)
#
#     with torch.no_grad():
#         output = model(image)
#         pred = output.argmax(dim=1).item()
#
#     return jsonify({'prediction': pred})

if __name__ == '__main__':
    app.run(debug=True)
