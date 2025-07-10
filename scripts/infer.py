import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import transforms
from PIL import Image
from model.resnet50 import resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(num_classes=10)
checkpoint_path = "Resnet_logs/best_model.pth"  
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
model.eval()

# 이미지 전처리
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),  
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
    )

img = Image.open("images/example.jpg").convert("RGB")  
input_tensor = transform(img).unsqueeze(0).to(device)

# 추론
with torch.no_grad():
    output = model(input_tensor)
    pred = torch.argmax(output, dim=1).item()

# 클래스 매핑
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

print(f"예측 결과: {classes[pred]}")
