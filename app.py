import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import numpy as np
import os

from siamese_model import SiameseNetwork

MODEL_PATH = "models/siamese_model.pth"
SUPPORT_DIR = "data/mvtec_ad/screw/train/good"
THRESHOLD = 0.6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SiameseNetwork().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model.forward_once(img_t)
    return emb.squeeze().cpu().numpy()

support_imgs = sorted(os.listdir(SUPPORT_DIR))[:5]
support_embeddings = [
    get_embedding(os.path.join(SUPPORT_DIR, im)) for im in support_imgs
]
support_embedding = np.mean(support_embeddings, axis=0)

def predict(img_path):
    emb = get_embedding(img_path)
    dist = np.linalg.norm(support_embedding - emb)
    label = "Normal" if dist < THRESHOLD else "Defect"
    return label, dist

root = tk.Tk()
root.title("Few-Shot Defect Detector")

img_label = Label(root)
img_label.pack()

result_label = Label(root, text="", font=("Arial", 14))
result_label.pack()

def open_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = Image.open(file_path)
    img_resized = img.resize((300, 300))
    tk_img = ImageTk.PhotoImage(img_resized)
    img_label.configure(image=tk_img)
    img_label.image = tk_img
    
    label, dist = predict(file_path)
    result_label.config(text=f"{label} (Distance: {dist:.3f})")

btn = Button(root, text="Select Image", command=open_image)
btn.pack()

root.mainloop()
