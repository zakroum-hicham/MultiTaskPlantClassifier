import numpy as np
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms
import os
from model import model

device = "cuda" if torch.cuda.is_available() else "cpu"


def file_to_image(data):
    image = Image.open(BytesIO(data)).convert('RGB')
    return image

def image_preparation(image):
    # image size
    image_size = 256

    # resize normalize transformations
    resize_normalize = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return resize_normalize(image)

def get_model(a):
    if a == "v1":
        cwd = os.getcwd()
        model_path = cwd+"\\models\\model-v1.pth"
        return torch.load(model_path, map_location=torch.device(device))
    else:
        return None
    
def predict(data,model_version="v1"):
    image = file_to_image(data)
    image = image_preparation(image)
    image = torch.unsqueeze(image,dim=0)

    model.eval()
    with torch.no_grad():
        image = image.to(device)

        model.load_state_dict(get_model(model_version))
        # forward
        leaf_logits, disease_logits = model(image)
        # leaf prediction
        leaf_predictions = torch.argmax(leaf_logits, dim=1)

        # disease prediction
        disease_predictions = torch.argmax(disease_logits, dim=1)

    return get_classes(leaf_predictions,disease_predictions)


def get_classes(leaf_type,disease_type):
    leaf_type = leaf_type.item()
    disease_type = disease_type.item()
    leaf_type_map = { 0:'Potato',  1:'Tomato',  2:'Pepper'}
    
    disease_type_map = {
            0:"Pepper__bell___Bacterial_spot",
            1: "Pepper__bell___healthy",
            2: "Potato___Early_blight",
            3: "Potato___healthy",
            4: "Potato___Late_blight",
            5: "Tomato__Target_Spot",
            6: "Tomato__Tomato_mosaic_virus",
            7: "Tomato__Tomato_YellowLeaf__Curl_Virus",
            8: "Tomato_Bacterial_spot",
            9: "Tomato_Early_blight",
            10 : "Tomato_healthy",
            11: "Tomato_Late_blight",
            12: "Tomato_Leaf_Mold",
            13: "Tomato_Septoria_leaf_spot",
            14: "Tomato_Spider_mites_Two_spotted_spider_mite",
        }
    return leaf_type_map[leaf_type],disease_type_map[disease_type]
    
    