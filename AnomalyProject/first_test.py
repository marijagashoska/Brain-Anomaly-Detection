# import torch
# from torchvision import transforms
# from PIL import Image
#
# # 1. Load the trained model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = torch.load('model.pth', map_location=device, weights_only=False)
#
# model.eval()  # IMPORTANT: set to eval mode
# model.to(device)
#
# # 2. Define the same validation transforms used during training
# val_transforms = transforms.Compose([
#     transforms.Resize((661, 661)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])
#
# # 3. Load and preprocess the image
# def predict_image(image_path):
#     image = Image.open(image_path).convert('RGB')  # Ensure 3 channels
#     input_tensor = val_transforms(image).unsqueeze(0).to(device)  # Add batch dimension
#
#     # 4. Forward pass
#     with torch.no_grad():
#         outputs = model(input_tensor)
#         _, predicted_idx = torch.max(outputs, 1)
#
#     return predicted_idx.item()
#
# # 5. Class names (same as used during training)
# class_names = ['trans_cerebellar', 'trans_thalamic', 'trans_ventricular']
#
# # Usage example:
# image_path = 'Patient01178_Plane3_4_of_4.png'
# predicted_class_idx = predict_image(image_path)
# print("Predicted class:", class_names[predicted_class_idx])
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torchvision import transforms
from PIL import Image
import os

# 1. Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('model.pth', map_location=device, weights_only=False)

model.eval()
model.to(device)

# 2. Validation transforms (same as training)
val_transforms = transforms.Compose([
    transforms.Resize((661, 661)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 3. Class names used during training
class_names = ['trans_cerebellar', 'trans_thalamic', 'trans_ventricular']

# 4. Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = val_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_idx = torch.max(outputs, 1)
    return class_names[predicted_idx.item()]

# 5. Process all images in a folder
def predict_folder_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            prediction = predict_image(image_path)
            print(f"{filename}: {prediction}")

# Example usage
folder_path = 'Orginal_train_images_to_959_661'  # Replace with your folder path
predict_folder_images(folder_path)
