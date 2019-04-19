import torch
import numpy as np

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


def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch):
    # Sets the learning rate to the initial LR decayed by 10 every lr_decay epochs
    lr = base_lr * (0.1 ** (epoch // lr_decay))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def matchmap_generate(image, text):
    """
    Generates Matchmap for a single image and text pair
    :param image: Single image embedding
    :param text: Single caption embedding
    :return: 3D tensor of matchmap
    """
    assert (image.dim() == 3)
    assert (text.dim() == 2)
    depth = image.size(0)
    height = image.size(1)
    width = image.size(2)

    seq_length = text.size(0)

    image_rep = image.view(depth, -1)
    match_map = torch.mm(text, image_rep)
    match_map = match_map.view(seq_length, height, width)

    return match_map


def score_from_matchmap(matchmap, score_type):
    """
    Generates similarity score from matchmap based on score_type
    :param matchmap: 3D matchmap
    :param score_type: Type of score you want
    :return: single tensor with similarity score
    """
    assert (matchmap.dim() == 3)
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


def score_function(image, text, score_type='Avg_Both'):
    """
    Generates score from image-text pair
    :param image: Single image embedding
    :param text: Single caption embedding
    :param score_type: Score type for score
    :return: single tensor score
    """
    matchmap = matchmap_generate(image, text)
    score = score_from_matchmap(matchmap, score_type)
    return score


def compute_matchmap_similarity_matrix(image_outputs, caption_outputs, score_type="Avg_Both"):
    """
    Generates a similarity score matrix for a given batch of image-caption data
    :param image_outputs: batch of image embeddings
    :param caption_outputs: batch of caption embeddings
    :param score_type: score type for similarity function
    :return: similarity matrix
    """
    assert(image_outputs.dim() == 4)
    assert(caption_outputs.dim() == 3)
    batch_size = image_outputs.size(0)

    sim_mat = torch.zeros(batch_size, batch_size, device=image_outputs.device)

    for image_idx in range(batch_size):
        for word_idx in range(batch_size):
            sim_mat[image_idx, word_idx] = score_function(image_outputs[image_idx], caption_outputs[word_idx], score_type)

    return sim_mat


def npairs_loss(image_outputs, caption_outputs, score_type='Avg_Both'):
    assert (image_outputs.dim() == 4)
    assert(caption_outputs.dim() == 3)

    sim_mat = compute_matchmap_similarity_matrix(image_outputs, caption_outputs, score_type)
    loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    n_imgs = image_outputs.size(0)


    for i in range(n_imgs):
        s_ii = torch.exp(sim_mat[i,i])
        s_ij = torch.sum(torch.exp(sim_mat[:,i]))
        s_ji = torch.sum(torch.exp(sim_mat[i,:]))

        l = torch.log(torch.div(s_ij, s_ii)) + torch.log(torch.div(s_ji, s_ii))

        loss = loss + l

    loss = loss / n_imgs
    return loss


def custom_loss(image_outputs, caption_outputs, score_type='Avg_Both',
                margin=0.1, sampler='hard'):
    assert (image_outputs.dim() == 4)
    assert (caption_outputs.dim() == 3)
    assert (type(sampler) == str)
    assert(type(score_type) == str)

    sim_mat = compute_matchmap_similarity_matrix(image_outputs, caption_outputs, score_type)
    n_imgs = image_outputs.size(0)
    loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)

    # C2I: Finding the best 2 images for each caption
    C2I_scores, C2I_ind = sim_mat.topk(2, 0)

    # I2C: Finding the best 2 captions for each image
    I2C_scores, I2C_ind = sim_mat.topk(2, 1)

    for i in range(n_imgs):
        if sampler == 'hard':
            # Hard Negative Triplet Mining
            img_impostor_index = C2I_ind[0][i]
            text_impostor_index = I2C_ind[i][0]

            if img_impostor_index == i:
                img_impostor_index = C2I_ind[1][i]
            if text_impostor_index == i:
                text_impostor_index = I2C_ind[i][1]


        else:
            img_impostor_index = i
            text_impostor_index = i
            # Random Triplet Sampling
            while img_impostor_index == i:
                img_impostor_index = np.random.randint(0, n_imgs)
            while text_impostor_index == i:
                text_impostor_index = np.random.randint(0, n_imgs)

        anchor_score = score_function(image_outputs[i], caption_outputs[i], score_type)
        image_imp_score = score_function(image_outputs[img_impostor_index], caption_outputs[i],score_type)
        text_imp_score = score_function(image_outputs[i], caption_outputs[text_impostor_index], score_type)

        text_imp = text_imp_score - anchor_score + margin
        image_imp = image_imp_score - anchor_score + margin

        if (text_imp.data > 0).all():
            loss = loss + text_imp

        if (image_imp.data > 0).all():
            loss = loss + image_imp

    loss = loss / n_imgs
    return loss


def calc_recalls(image_outputs, caption_outputs, score_type):

    sim_mat = compute_matchmap_similarity_matrix(image_outputs, caption_outputs, score_type)
    batch_size = sim_mat.size(0)

    # torch.topk() returns the k largest elements of a given input tensor along a given dimension
    # C2I: Finding the best k images for each caption
    C2I_scores, C2I_ind = sim_mat.topk(10, 0)

    # I2C: Finding the best k captions for each image
    I2C_scores, I2C_ind = sim_mat.topk(10, 1)

    # C_rk : Caption recall at k, and  I_rk : Image recall at k
    C_r1 = AverageMeter()
    C_r5 = AverageMeter()
    C_r10 = AverageMeter()
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()

    for i in range(batch_size):
        C_foundind = -1
        I_foundind = -1

        for ind in range(10):
            # Is the image in top 10 of the caption?
            if C2I_ind[ind, i] == i:
                I_foundind = ind

            # Is the caption in top 10 of the image?
            if I2C_ind[i, ind] == i:
                C_foundind = ind

        # Recall at 1
        # If the caption has been found at index 0
        if C_foundind == 0:
            C_r1.update(1)  # Found in top 1
        else:
            C_r1.update(0)

        # If the image has been found at index 0
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)

        # Recall at 5
        # If the caption has been found in the top 5
        if C_foundind >= 0 and C_foundind < 5:
            C_r5.update(1)  # Found in top 5
        else:
            C_r5.update(0)

        # If the image has been found in the top 5
        if I_foundind >=0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)

        # Recall at 10
        # If the caption has been found in the top 10
        if C_foundind >= 0 and C_foundind < 10:
            C_r10.update(1)  # Found in top 10
        else:
            C_r10.update(0)

        # If the image has been found in the top 10
        if I_foundind >=0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)

    # Create a dictionary of recall scores
    recalls = {'C_r1': C_r1.avg, 'C_r5': C_r5.avg, 'C_r10': C_r10.avg,
               'I_r1': C_r1.avg, 'I_r5': I_r5.avg, 'I_r10': I_r10.avg}

    return recalls
