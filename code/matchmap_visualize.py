import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
from torchvision import transforms
import torch.utils.data as data


import argparse
import numpy as np
import math
import os
import matplotlib.pyplot as plt

from utils import *
from models import *
from data_loader import get_loader

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--resume_filename", type=str, default="",
                    help="Path to checkpoint file")
parser.add_argument("--cnn_model", type=str, default="vgg",
                    help="CNN Architecture used")
args = parser.parse_args()

caption_model = LSTMBranch(batch_size = 64)

if args.cnn_model == 'vgg':
    image_model = VGG19(pretrained=True)
    print('VGG Model Loaded')
else:
    image_model = ResNet50(pretrained=True)
    print('ResNet model Loaded')



if args.resume_filename:
    if os.path.isfile(args.resume_filename):
        print("=> Loading Checkpoint '{}'".format(args.resume_filename))
        checkpoint = torch.load(args.resume_filename)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        image_model.load_state_dict(checkpoint['image_model'])
        caption_model.load_state_dict(checkpoint['caption_model'])
        print("Loaded checkpoint '{}' (epoch {})".format(args.resume_filename, checkpoint['epoch']))


    else:
        print(" => No checkpoint found at '{}'".format(args.resume_filename))

else:
    print("Checkpoint file not provided!")

if torch.cuda.is_available() and args.use_gpu == True:
    print("Loading models onto GPU to accelerate training")
    image_model = image_model.cuda()
    caption_model = caption_model.cuda()

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

data_loader_val = get_loader(transform=transform,
                             mode='val',
                             batch_size=args.batch_size,
                             vocab_from_file=True)


