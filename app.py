import os
import urllib.request
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REGIONS = {
    0: "American",
    1: "African",
    2: "Asian",
    3: "Indian",
    4: "European"
}

def load_trained_model(weights_path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def main():
    model_path = "region_model_optimized.pth"
    
    try:
        model = load_trained_model(model_path)
        print("Model muvaffaqiyatli yuklandi!")
    except Exception as e:
        print(f"Xatolik: Model yuklanmadi. Fayl yo'lini tekshiring: {e}")
        return

    # Haar Cascade faylini local papkaga avtomatik yuklab olish
    cascade_path = "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        print("Yuzni aniqlash fayli yuklanmoqda...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, cascade_path)

    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Kamerani ochib bo'lmadi!")
        return

    print("Kamera ishga tushdi! Dasturdan chiqish uchun 'q' tugmasini bosing.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)
            
            input_tensor = transform(pil_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                region_name = REGIONS[predicted_idx.item()]
                score = confidence.item() * 100

            label = f"{region_name}: {score:.1f}%"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Face Region Classifier", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()