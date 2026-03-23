import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class PairDataset(Dataset):
    def __init__(self, good_dir, defect_dir):
        self.good_images = [os.path.join(good_dir, f) for f in os.listdir(good_dir)]
        self.defect_images = []

        for root, _, files in os.walk(defect_dir):
            if "good" not in root:
                for f in files:
                    self.defect_images.append(os.path.join(root, f))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.good_images)

    def __getitem__(self, idx):
        img1_path = self.good_images[idx]
        img1 = Image.open(img1_path).convert("RGB")

        if random.random() > 0.5:
            img2_path = random.choice(self.good_images)
            label = 1
        else:
            img2_path = random.choice(self.defect_images)
            label = 0

        img2 = Image.open(img2_path).convert("RGB")

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

