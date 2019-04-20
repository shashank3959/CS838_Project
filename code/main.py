import argparse
import os
import time
from torchvision import transforms
from data_loader import get_loader
import numpy as np
import shutil
import torch.utils.data as data
import torch
import models
from models_train import train
from models_train import validate
from models import VGG19, LSTMBranch, ResNet50
import math
from utils import adjust_learning_rate
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--lr_decay', default=10, type=int, metavar='LRDECAY',
                    help='Divide the learning rate by 10 every lr_decay epochs')

parser.add_argument("--n_epochs", type=int, default=10,
                    help="number of maximum training epochs. -1 default refers to infinite")

parser.add_argument("--margin", type=float, default=0.1,
                    help="Margin parameter for triplet loss")

parser.add_argument("--use_gpu", type=bool, default=True,
                    help="Use GPU to accelerate training")

parser.add_argument('--loss_type', default='triplet', type=str,
                    help='kind of loss function to be implemented')

parser.add_argument('--score_type', type=str, default='Avg_Both',
                    help='Metric used to compute score.')

parser.add_argument("--sampler", type=str, default='hard',
                    help="Sampling strategy")

parser.add_argument("--optim", type=str, default="sgd",
                    help="training optimizer", choices=["sgd", "adam"])

parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 100)')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--weight_decay', '--wd', default=1e-7, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument("--crop_size", type=int, default=224,
                    help="size for randomly cropping images")

parser.add_argument('--name', default='Two_Branch_Image_Sentence', type=str,
                    help='name of experiment')

parser.add_argument('--minimum_gain', type=float, default=5e-1, metavar='N',
                    help='minimum performance gain for a model to be considered better. (default: 5e-1)')

parser.add_argument('--no_gain_stop', type=int, default=10, metavar='N',
                    help='number of epochs used to perform early stopping based on validation performance (default: 10)')

parser.add_argument("--cnn_model", type=str, default='vgg',
                    help="CNN Model")

parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint of best model (default: none)')


def main(args):
    # Parsing command line arguments
    print("Process %s, running on %s: starting (%s)" % (
        os.getpid(), os.name, time.asctime()))

    if args.cnn_model == 'vgg':
        image_model = VGG19(pretrained=True)
    else:
        image_model = ResNet50(pretrained = True)

    caption_model = LSTMBranch()

    if torch.cuda.is_available() and args.use_gpu == True:
        print("Loading models onto GPU to accelerate training")
        image_model = image_model.cuda()
        caption_model = caption_model.cuda()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
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

    # Load saved model
    start_epoch, best_loss = load_checkpoint(image_model, caption_model, args.resume)

    # Get the learnable parameters
    image_trainables = [p for p in image_model.parameters() if p.requires_grad]
    caption_trainables = [p for p in caption_model.parameters() if p.requires_grad]
    params = image_trainables + caption_trainables

    # optimizer = torch.optim.Adam(params=params, lr=0.01)
    optimizer = torch.optim.SGD(params=params, lr=args.lr, momentum=0.9)

    total_train_step = math.ceil(
        len(data_loader_train.dataset.caption_lengths) / data_loader_train.batch_sampler.batch_size)
    # print("Total number of training steps are :", total_train_step)

    print("========================================================")
    print("Loss Type: ", args.loss_type)
    if args.loss_type == 'triplet':
        print("Sampling strategy: ", args.sampler)
        print("Margin for triplet loss: ", args.margin)
    print("Learning Rate: ", args.lr)
    print("Score Type for similarity: ", args.score_type)
    print("========================================================")

    epoch = start_epoch
    best_epoch = start_epoch

    while (epoch - best_epoch) < args.no_gain_stop and (epoch <= args.n_epochs):

        adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch)
        print("========================================================")
        print("Epoch: %d Training starting" % epoch)
        print("Learning rate : ", get_lr(optimizer))
        train_loss = train(data_loader_train, data_loader_val, image_model,
                              caption_model, args.loss_type, optimizer, epoch,
                              args.score_type, args.sampler, args.margin,
                              total_train_step, args.batch_size, args.use_gpu)
        print('---------------------------------------------------------')
        print("Epoch: %d Validation starting" % epoch)
        val_loss = validate(caption_model, image_model, data_loader_val,
                            epoch, args.loss_type, args.score_type, args.sampler,
                            args.margin, args.use_gpu)
        print("Epoch: ", epoch)
        print("Training Loss: ", float(train_loss.data))
        print("Validation Loss: ", float(val_loss.data))

        print("========================================================")

        save_checkpoint({
            'epoch': epoch,
            'best_loss': min(best_loss, val_loss),
            'image_model': image_model.state_dict(),
            'caption_model': caption_model.state_dict()
        }, val_loss < best_loss)
        print("Saved Checkpoint!")
        if (val_loss) < best_loss:
            best_epoch = epoch
            best_loss = val_loss

        epoch += 1

    print("Back to main")
    resume_filename = 'runs/%s/' % (args.name) + 'model_best.pth.tar'
    if os.path.exists(resume_filename):

        epoch, best_loss1 = load_checkpoint(image_model, caption_model, args.resume)
        val_loss1 = validate(caption_model, image_model, data_loader_val,
                                epoch, args.loss_type, args.score_type, args.sampler,
                                args.margin, args.use_gpu)
        print("========================================================")
        print("========================================================")
        print("Final Loss : ", float(val_loss1.data))
        print("========================================================")
        print("========================================================")

    else:
        resume_filename = 'runs/%s/' % (args.name) + 'checkpoint.pth.tar'
        print("Using last run epoch.")
        epoch, best_loss1 = load_checkpoint(image_model, caption_model, args.resume)
        val_loss1 = validate(caption_model, image_model, data_loader_val,
                             epoch, args.loss_type, args.score_type, args.sampler,
                             args.margin, args.use_gpu)
        print("========================================================")
        print("========================================================")
        print("Final Loss : ", float(val_loss1.data))
        print("========================================================")
        print("========================================================")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (args.name)

    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = directory + filename
    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.name) + 'model_best.pth.tar')


def load_checkpoint(image_model, caption_model, resume_filename):
    start_epoch = 1
    if args.loss_type == 'triplet':
        best_loss = 2 * args.margin
    else:
        best_loss = 1.0

    if resume_filename:
        if os.path.isfile(resume_filename):
            print("=> Loading Checkpoint '{}'".format(resume_filename))
            checkpoint = torch.load(resume_filename)
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            image_model.load_state_dict(checkpoint['image_model'])
            caption_model.load_state_dict(checkpoint['caption_model'])
            print("Loaded checkpoint '{}' (epoch {})".format(resume_filename, checkpoint['epoch']))

        else:
            print(" => No checkpoint found at '{}'".format(resume_filename))

    return start_epoch, best_loss


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
