import torch
import numpy as np


def matchmap_generate(image, text):
    assert (image.dim() == 3)
    assert(text.dim() == 2)
    depth = image.size(0)
    height = image.size(1)
    width = image.size(2)

    seq_length = text.size(0)

    image_rep = image.view(depth, -1)
    match_map = torch.mm(text, image_rep)
    match_map = match_map.view(seq_length, height, width)

    return match_map


def compute_similarity_score(matchmap, score_type):
    assert(matchmap.dim() == 3)
    if score_type == 'Avg_Both':
        return matchmap.mean()
    elif score_type == 'Max_Img':
        max_height, _ = torch.max(matchmap, 1)
        max_image, _ = torch.max(matchmap, 1)
        return max_image.mean()
    elif score_type == 'Max_Text':
        max_text, _ = torch.max(matchmap, 0)
        return max_text.mean()
    else:
        raise ValueError


def custom_loss(image_output, text_output, score_type = 'Max_Img', margin = 1):

    assert(image_output.dim() == 4)
    assert(text_output.dim() == 3)

    n_imgs = image_output.size(0)

    loss = torch.zeros(1, device=image_output.device, requires_grad=True)

    for i in range(n_imgs):
        img_impostor_index = i
        text_impostor_index = i

        # Create impostor index
        while img_impostor_index == i:
            img_impostor_index = np.random.randint(0,n_imgs)
        while text_impostor_index == i:
            text_impostor_index = np.random.randint(0,n_imgs)

        anchor_score = compute_similarity_score(matchmap_generate(image_output[i], text_output[i]), score_type)
        image_imp_score = compute_similarity_score(matchmap_generate(image_output[img_impostor_index], text_output[i]), score_type)
        text_imp_score = compute_similarity_score(matchmap_generate(image_output[i], text_output[text_impostor_index]), score_type)

        text_imp = text_imp_score - anchor_score + margin
        image_imp = image_imp_score - anchor_score + margin

        if (text_imp.data > 0).all():
            loss = loss + text_imp

        if (image_imp.data > 0).all():
            loss = loss + image_imp

    loss = loss/n_imgs
    return loss


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
