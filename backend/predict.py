import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

# Load classes
with open("models/classes.json", "r") as f:
    classes = json.load(f)

# Load model
model = efficientnet_b0(weights=None)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(classes))

model.load_state_dict(
    torch.load("models/emstrap_best.pth", map_location="cpu")
)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# Enter image path
image_path = input("Enter image path: ")

img = Image.open(image_path).convert("RGB")
plt.imshow(img)
plt.axis("off")
plt.title("Input Image")
plt.show()
img = transform(img).unsqueeze(0)

with torch.no_grad():
    outputs = model(img)
    probs = torch.softmax(outputs, dim=1)[0]

# Top 3 predictions
top_probs, top_idxs = torch.topk(probs, 3)

prediction = classes[top_idxs[0].item()]
confidence = top_probs[0].item() * 100

    # Get top 3 predictions
top_probs, top_idxs = torch.topk(probs, 3)

# Severity estimation
if prediction=="fire":
    severity = "CRITICAL" if confidence>90 else "HIGH"

elif prediction=="accident":
    severity = "HIGH" if confidence>85 else "MEDIUM"

elif prediction=="human_accident":
    severity = "MEDIUM"

elif prediction=="human_burn":
    severity = "CRITICAL"

else:
    severity = "NONE"

# Display image with prediction
display_img = Image.open(image_path).convert("RGB")

plt.figure(figsize=(8,6))
plt.imshow(display_img)
plt.axis("off")

plt.title(
    f"Prediction : {prediction}\n"
    f"Confidence : {confidence:.2f}%\n"
    f"Severity : {severity}",
    fontsize=14
)

plt.show()
print("\nTop Predictions:")
for p, i in zip(top_probs, top_idxs):
    print(f"{classes[i]} : {p.item()*100:.2f}%")

if top_probs[0].item() < 0.75:
    print("\n⚠️ Low confidence prediction.")
    print("The image may be ambiguous or outside the training distribution.")