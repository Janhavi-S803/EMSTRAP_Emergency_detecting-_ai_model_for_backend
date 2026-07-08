import torch
import json
from pathlib import Path
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("models/classes.json") as f:
    classes = json.load(f)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

dataset = datasets.ImageFolder(
    "dataset/validation",
    transform=transform
)

loader = DataLoader(dataset, batch_size=32)

model = efficientnet_b0()
model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features,
    len(classes)
)

model.load_state_dict(
    torch.load("models/emstrap_best.pth")
)

model.to(device)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for x,y in loader:
        x = x.to(device)
        pred = model(x).argmax(1).cpu()
        y_true.extend(y.numpy())
        y_pred.extend(pred.numpy())

print(classification_report(
    y_true,
    y_pred,
    target_names=classes
))

cm = confusion_matrix(y_true,y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=classes,
    yticklabels=classes
)
plt.savefig("reports/confusion_matrix.png")
print("saved")