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


def caption_list_gen(caption):
    # Returns a list of captions
    caption_list = []
    for i in range(len(caption[0])):
        new_caption = []
        for j in range(len(caption)):
            new_caption.append(caption[j][i])
        caption_list.append(new_caption)
    return caption_list


def get_data(batch_size, fetch_mode='retrieval'):
    # Returns tensors of image and caption glove embeddings to be fed to models
    # Also returns textual captions for visualization
    data_loader_val = get_loader(transform=transform,
                                 mode='val',
                                 batch_size=batch_size,
                                 vocab_from_file=True,
                                 fetch_mode='retrieval')

    indices = data_loader_val.dataset.get_indices()
    new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
    data_loader_val.batch_sampler.sampler = new_sampler

    for batch in data_loader_val:
        image_tensor, caption_glove_tensor, captions = batch[0], batch[1], batch[2]
        break

    caption_list = caption_list_gen(captions)

    return image_tensor, caption_glove_tensor, caption_list


def load_model(model_path = 'model_best.pth.tar', map_location='cpu'):
    # Load model based on model path provided
    image_model = VGG19(pretrained=True)
    checkpoint = torch.load(model_path, map_location)
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    image_model.load_state_dict(checkpoint['image_model'])
    caption_model.load_state_dict(checkpoint['caption_model'])
    print("Loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))
    return image_model, caption_model


def rgb2gray(rgb):
    # Convert color numpy image to grayscale
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def tensor2img(tensor_image):
    # Convert each tensor image to numpy image
    img = tensor_image.permute(1,2,0)
    color_img = img.numpy()
    bw_img = rgb2gray(color_img)

    return color_img, bw_img


def gen_matchmap(image_model, caption_model, image_tensor, caption_tensor):
    # Generates a stack of matchmaps of length batch size
    image_op = image_model(image_tensor)
    caption_op = caption_model(caption_tensor)

    n_imgs = image_op.size(0)
    matchmap_list = []

    for i in range(n_imgs):
        matchmap = matchmap_generate(image_op[i], caption_op[i])
        mm = matchmap.detach().numpy()
        matchmap_list.append(mm)

    return matchmap_list


def gen_masks(matchmap_list, image_tensor, caption_list, index = 0):
    # Creates a list of matchmaps for each image.
    n_imgs = len(matchmap_list)

    assert n_imgs >= index

    target_image = image_tensor[index-1]
    target_caption = caption_list[index-1]
    target_matchmap = matchmap_list[index-1]

    color_img, bw_img = tensor2img(target_image)

    mask_list = []
    for mmap in target_matchmap:
        mask = cv2.resize(mmap, dsize=(224, 224))
        mask_list.append(mask)

    mask_list = np.stack(mask_list, axis=0)

    return color_img, bw_img, mask_list, target_caption


def gen_results(color_img, bw_img, mask_list, caption, name, save_flag = False):
    color_img = (color_img - np.amin(color_img)) / np.ptp(color_img)
    # color_img = 255*(color_img - np.amin(color_img))/np.ptp(color_img).astype(int)
    plt.imshow(color_img)
    plt.title('Original Image')
    plt.axis('off')
    plt.show()
    orig_name = name + 'original.png'
    if save_flag:
        plt.imsave(orig_name, color_img)

    fig = plt.figure(figsize=(30, 10), facecolor='white')

    columns = len(mask_list) + 1
    rows = 1

    for id in range(len(mask_list)):
        cap_word = caption[id]
        mask = cv2.resize(mask_list[id], dsize=(224, 224))
        if not cap_word in ['<start>', '<end>', ',', '.', '<unk>']:
            fig.add_subplot(rows, columns, id+1)
            plt.imshow(bw_img)
            plt.imshow(mask, cmap='jet', alpha=0.5)
            plt.title(cap_word, fontdict={'fontsize': 20})
            plt.axis('off')

    plt.show()
    result_name = name + 'results.png'
    if save_flag:
        fig.savefig(result_name)
