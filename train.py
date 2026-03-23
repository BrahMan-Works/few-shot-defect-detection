import torch
from torch.utils.data import DataLoader
from siamese_model import SiameseNetwork
from contrastive_loss import ContrastiveLoss
from dataset import PairDataset
import torch.optim as optim
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

good_dir = "data/mvtec_ad/screw/train/good"
defect_dir = "data/mvtec_ad/screw/test"

dataset = PairDataset(good_dir, defect_dir)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = SiameseNetwork().to(device)
criterion = ContrastiveLoss(margin=1.0)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for img1, img2, label in tqdm(loader):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        out1, out2 = model(img1, img2)
        loss = criterion(out1, out2, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader)}")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/siamese_model.pth")
print("Model saved.")

