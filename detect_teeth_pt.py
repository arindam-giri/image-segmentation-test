import tensorflow as tf
import numpy as np
import cv2
from huggingface_hub import from_pretrained_keras
import os
from PIL import Image

# Load the model
# try:
#     model = from_pretrained_keras("SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net")
# except:
model = tf.keras.models.load_model("dental_xray_seg.h5")

# Hardcoded image paths
examples = ["107.png", "108.png", "109.png"]

# Output folder
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

def load_image(image_file):
    img = Image.open(image_file)
    return img

def convert_one_channel(img):
    # Some images have 3 channels, although they are grayscale images
    if len(img.shape) > 2:
        img = img[:, :, 0]
        return img
    else:
        return img

def convert_rgb(img):
    # Some images have 3 channels, although they are grayscale images
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img
    else:
        return img

def process_image(image_file):
    img = cv2.imread(image_file)
    img_cv = convert_one_channel(img)
    img_cv = cv2.resize(img_cv, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    img_cv = np.float32(img_cv / 255)
    img_cv = np.reshape(img_cv, (1, 512, 512, 1))
    
    prediction = model.predict(img_cv)
    predicted = prediction[0]
    predicted = cv2.resize(predicted, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    mask = np.uint8(predicted * 255)
    _, mask = cv2.threshold(mask, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), dtype=np.float32)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = cv2.drawContours(convert_rgb(img), cnts, -1, (255, 0, 0), 3)
    
    return output

def save_image(output, image_file):
    output_path = os.path.join(output_folder, os.path.basename(image_file))
    cv2.imwrite(output_path, output)
    print(f"Saved output image to {output_path}")

def main():
    for example in examples:
        print(f"Processing {example}...")
        output = process_image(example)
        save_image(output, example)
        print(f"Done processing {example}.")

if __name__ == "__main__":
    main()