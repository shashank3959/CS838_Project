import numpy as np
import torch


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


# # Helper code to check functioning of matchmap_generate()
#
# image = torch.randn(256, 14, 14)
# text = torch.randn(10, 256)
#
# matchmap = matchmap_generate(image, text)
#
# print("1. Image shape: ", image.shape)
# print("2. Text shape: ", text.shape)
# print("3. Matchmap shape: ", matchmap.shape)
# s = compute_similarity_score(matchmap, 'Max_Text')
#
# print("4. Similarity Score: ",s)


