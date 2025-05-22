from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import requests

# Flask 애플리케이션 생성
app = Flask(__name__)

# 모델 아키텍처와 클래스 수 설정
num_classes = 31  # 실제 클래스 수로 변경
model = models.resnet18(weights=None)

# fc 레이어 수정 (체크포인트에 맞춰 수정)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 128),  # 중간 레이어
    torch.nn.ReLU(),
    torch.nn.Linear(128, num_classes)  # 최종 레이어
)

# 체크포인트 경로
checkpoint_path = './best_model_last_4.pth'

def load_model():
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # 모델의 state_dict 업데이트
    model.load_state_dict(checkpoint, strict=False)  # strict=False로 변경하여 키 불일치를 무시
    model.eval()  # 평가 모드로 설정
    return model

# 이미지 전처리 함수
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # RGB로 변환
    return transform(image).unsqueeze(0)  # 배치 차원 추가

# 예측 함수
def get_prediction(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()
    
# 루트 엔드포인트
@app.route('/')
def home():
    return "Welcome to the model serving API!"

# 이미지 예측 엔드포인트 - POST 방식
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image_bytes = file.read()
    image_tensor = transform_image(image_bytes)
    prediction = get_prediction(image_tensor)
    
    print(f'Prediction: {prediction}')
    return jsonify({'prediction': prediction})

# 이미지 예측 엔드포인트 - GET 방식 (URL의 이미지 경로 사용)
@app.route('/predict', methods=['GET'])
def predict_get():
    image_path = request.args.get('image_path')
    if not image_path:
        return jsonify({'error': 'No image path provided'}), 400

    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        image_tensor = transform_image(image_bytes)
        prediction = get_prediction(image_tensor)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 모델 로드
    model = load_model()
    # 서버 실행
    app.run(host='0.0.0.0', port=5000)