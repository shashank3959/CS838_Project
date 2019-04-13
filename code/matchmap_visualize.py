import torch
from torchvision import transforms
import torch.utils.data as data
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from models import *
from data_loader import get_loader


image_model = VGG19(pretrained=True)
caption_model = LSTMBranch()

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def load_model(model_path = 'model_best.pth.tar', map_location='cpu'):
    image_model = VGG19(pretrained=True)
    checkpoint = torch.load(model_path, map_location)
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    image_model.load_state_dict(checkpoint['image_model'])
    caption_model.load_state_dict(checkpoint['caption_model'])
    print("Loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))
    return image_model, caption_model


def gen_matchmap(image_model, caption_model, image_tensor, caption_tensor):
    image_op = image_model(image_tensor)
    caption_op = caption_model(caption_tensor)

    matchmap = matchmap_generate(image_op[0], caption_op[0])
    mm = matchmap.detach().numpy()

    return mm


def gen_masks(image_model, caption_model, image_tensor, caption_tensor):
    matchmap = gen_matchmap(image_model, caption_model, image_tensor, caption_tensor)
    I = image_tensor[0].permute(1, 2, 0)
    I = I.numpy()
    Img = rgb2gray(I)
    mask_list = []
    for mmap in matchmap:
        mask = cv2.resize(mmap, dsize=(224, 224))
        mask_list.append(mask)

    mask_list = np.stack(mask_list, axis=0)

    return Img, I, mask_list