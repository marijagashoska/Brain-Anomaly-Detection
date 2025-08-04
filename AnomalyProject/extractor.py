# from PIL import Image
# import pytesseract
# import re
#
# # Point to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Lenovo\Documents\fakultetski\OneId Project\Tesseract-OCR\Tesseract-OCR\tesseract.exe"
#
# def extract_trimester_label(image_path):
#     # Load the image
#     image = Image.open(image_path)
#     width, height = image.size
#
#     # Crop top-right corner — adjust percentages if needed
#     cropped = image.crop((int(width * 0.6), 0, width, int(height * 0.25)))
#
#     # Show cropped area for debugging
#     cropped.show()  # Opens image viewer
#     # Optional: Save instead
#     # cropped.save("cropped_preview.png")
#
#     # OCR the cropped region
#     text = pytesseract.image_to_string(cropped,config='--psm 6')
#     print("OCR Output:\n", text)
#
#     # Search for trimester labels like "2 Trim", "2+3 Trim"
#     match = re.findall(r'(\d(?:\.\s*)?(?:\+\s*\d(?:\.\s*)?)?\s*trim\.?)', text.lower())
#
#     return match[0] if match else None
#
# # Example usage
# if __name__ == "__main__":
#     image_path = "Patient00305_Plane3_3_of_5.png"  # Use your uploaded file path
#     label = extract_trimester_label(image_path)
#     if label:
#         print(f"\n✅ Detected trimester label: {label}")
#     else:
#         print("\n❌ No trimester label found.")
#
#
import os
import shutil
from PIL import Image
import pytesseract
import re

# Point to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Lenovo\Documents\fakultetski\OneId Project\Tesseract-OCR\Tesseract-OCR\tesseract.exe"

def extract_trimester_label(image_path):
    image = Image.open(image_path)
    width, height = image.size
    cropped = image.crop((int(width * 0.6), 0, width, int(height * 0.25)))

    text = pytesseract.image_to_string(cropped, config='--psm 6')
    match = re.findall(r'(\d(?:\.\s*)?(?:\+\s*\d(?:\.\s*)?)?\s*trim\.?)', text.lower())
    return match[0].replace(" ", "").replace(".", "") if match else None

def sort_images_by_trimester(source_folders, output_root="output"):
    for folder in source_folders:
        for filename in os.listdir(folder):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(folder, filename)
                label = extract_trimester_label(image_path)

                if label:
                    label_clean = label.lower().replace("trim", "").replace(".", "").replace(" ", "")
                    if label_clean in ["2", "3", "2+3"]:
                        dest_folder = os.path.join(output_root, f"{label_clean}_trim")
                        os.makedirs(dest_folder, exist_ok=True)
                        shutil.copy(image_path, os.path.join(dest_folder, filename))
                        print(f"✅ Moved '{filename}' to '{dest_folder}'")
                    else:
                        print(f"⚠️ Unknown label '{label}' in file: {filename}")
                else:
                    print(f"❌ No trimester label found in: {filename}")

# Example usage
if __name__ == "__main__":
    source_folders = [
        # "dataset/train/trans_cerebellar",
        "dataset/train/trans_thalamic",
        # "dataset/train/other_folder"
    ]

    sort_images_by_trimester(source_folders, output_root="sorted")
