import argparse
import os
import time
from torchvision import transforms
from data_loader import get_loader
import numpy as np
import torch.utils.data as data
import torch
import models
from models_train import train
from models_train import validate
from models import VGG19, LSTMBranch
import math
from utils import adjust_learning_rate

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='',
                    help="training data json")
parser.add_argument("--data-val", type=str, default='',
                    help="validation data json")
parser.add_argument("--exp-dir", type=str, default="",
                    help="directory to dump experiments")
parser.add_argument("--resume", action="store_true", dest="resume",
                    help="load from exp_dir if True")
parser.add_argument("--optim", type=str, default="sgd",
                    help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay', default=40, type=int, metavar='LRDECAY',
                    help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-7, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--n_epochs", type=int, default=50,
                    help="number of maximum training epochs")
parser.add_argument("--n_print_steps", type=int, default=100,
                    help="number of steps to print statistics")
parser.add_argument("--image-model", type=str, default="VGG16",
                    help="image model architecture", choices=["VGG16"])
parser.add_argument("--pretrained-image-model", action="store_true",
                    dest="pretrained_image_model", help="Use an image network pretrained on ImageNet")
parser.add_argument("--margin", type=float, default=1.0, help="Margin parameter for triplet loss")
parser.add_argument("--crop_size", type=int, default=224, help="size for randomly cropping images")
parser.add_argument("--use_gpu", type=bool, default=True, help="Use GPU to accelerate training")


# resume from a checkpoint? - add code here
def main(args):
    # Parsing command line arguments
    print("Process %s, running on %s: starting (%s)" % (
        os.getpid(), os.name, time.asctime()))

    # VGG-19 based model for Images
    image_model = VGG19(pretrained=args.pretrained_image_model)

    # LSTM model for caption glove embeddings
    caption_model = LSTMBranch(args.batch_size)
    # Move to GPU if CUDA is available
    if torch.cuda.is_available() and args.use_gpu == True:
        print("Loading models onto GPU to accelerate training")
        image_model = image_model.cuda()
        caption_model = caption_model.cuda()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])

    # Obtain the data loader (from file). Note that it runs much faster than before!
    data_loader_train = get_loader(transform=transform,
                                   mode='train',
                                   batch_size=args.batch_size,
                                   vocab_from_file=True)

    data_loader_val = get_loader(transform=transform,
                                 mode='val',
                                 batch_size=args.batch_size,
                                 vocab_from_file=True)

    # Get the learnable parameters
    params = list(image_model.parameters()) + list(caption_model.parameters())

    # Define the optimizer
    # optimizer = torch.optim.Adam(params=params, lr=0.01)
    optimizer = torch.optim.SGD(params=params, lr=0.01, momentum=0.9)

    total_train_step = math.ceil(
        len(data_loader_train.dataset.caption_lengths) / data_loader_train.batch_sampler.batch_size)
    # print("Total number of training steps are :", total_train_step)

    for epoch in range(1, args.n_epochs + 1):
        adjust_learning_rate(optimizer, epoch)
        print("========================================================")
        print("Epoch: %d starting" % (epoch))
        print("Learning rate : ", get_lr(optimizer))
        train_loss = train(data_loader_train, data_loader_val, image_model, caption_model, optimizer, epoch,
                           total_train_step, args.batch_size, args.use_gpu)
        print("---------------------------------------------------------")
        print("Epoch: ", epoch)
        print("Training Loss: ", train_loss)
        validate(caption_model, image_model, data_loader_val, args.use_gpu)
        print("========================================================")

    print("Back to main")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
