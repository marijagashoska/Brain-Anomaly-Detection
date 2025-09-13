import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torchvision import transforms
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('model.pth', map_location=device, weights_only=False)

model.eval()
model.to(device)

val_transforms = transforms.Compose([
    transforms.Resize((661, 661)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ['trans_cerebellar', 'trans_thalamic', 'trans_ventricular']

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = val_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_idx = torch.max(outputs, 1)
    return class_names[predicted_idx.item()]

def predict_folder_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            prediction = predict_image(image_path)
            print(f"{filename}: {prediction}")


folder_path = 'Orginal_train_images_to_959_661'
predict_folder_images(folder_path)
