import torch
# from utils import resize_image
from PIL import Image
from ultralytics import YOLO

device = torch.device("cuda")
model = YOLO('../model/FastSAM')

raw_image = Image.open("./env/image/dog1.jpg")
raw_image
# Resize
