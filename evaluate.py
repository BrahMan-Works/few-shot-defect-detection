import torch
import torch.nn.functional as F
from siamese_model import SiameseNetwork
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("models/siamese_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.forward_once(img)
    return embedding.squeeze().cpu().numpy()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

support_folder = "data/mvtec_ad/screw/train/good"
support_images = sorted(os.listdir(support_folder))[:5]

support_embeddings = []
for img_name in support_images:
    emb = get_embedding(os.path.join(support_folder, img_name))
    support_embeddings.append(emb)

support_embedding = np.mean(support_embeddings, axis=0)

test_folder = "data/mvtec_ad/screw/test"

y_true = []
y_pred = []

threshold = 0.6
for root, _, files in os.walk(test_folder):
    for file in tqdm(files):
        img_path = os.path.join(root, file)
        emb = get_embedding(img_path)
        dist = np.linalg.norm(support_embedding - emb)

        label = 0 if "good" in root else 1
        pred = 0 if dist < threshold else 1

        y_true.append(label)
        y_pred.append(pred)

with open("results/final_metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy_score(y_true, y_pred)}\n")
    f.write(f"Precision: {precision_score(y_true, y_pred)}\n")
    f.write(f"Recall: {recall_score(y_true, y_pred)}\n")
    f.write(f"F1 Score: {f1_score(y_true, y_pred)}\n")
    f.write(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}\n")

