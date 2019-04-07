import time
import torch.utils.data as data
from utils import matchmap_generate, compute_similarity_score, custom_loss
import torch

def train(data_loader_train, image_model, caption_model, epoch, total_train_step, batch_size, use_gpu=False, start_step=1, start_loss=0.0):
    """Train model for exactly one epoch using the parameters given"""

    image_model.train()
    caption_model.train()

    total_loss = start_loss

    start_time = time.time()

    loss_scores = list()

    #for i_step in range(start_step, total_train_step + 1):
    for i_step in range(1, 50):
        print("i_step is:", i_step)

        indices = data_loader_train.dataset.get_indices()
        # Create a batch sampler to retrieve a batch with the sampled indices
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader_train.batch_sampler.sampler = new_sampler

        # Obtain the batch
        for batch in data_loader_train:
            image_ip, caption_glove_ip = batch[0], batch[1]
            break

        # Move to GPU if CUDA is available
        if torch.cuda.is_available() and use_gpu==True:
            image_ip = image_ip.cuda()
            caption_glove_ip = caption_glove_ip.cuda()

        image_output = image_model(image_ip)
        caption_glove_output = caption_model(caption_glove_ip, use_gpu)

        sim_scores = list()

        for sample in range(batch_size):
            mmap = matchmap_generate(image_output[sample], caption_glove_output[sample])
            score = compute_similarity_score(mmap, "Max_Img")
            sim_scores.append(score)

        loss = custom_loss(image_output, caption_glove_output)
        print("Step: %d, and loss: %0.4f" % (i_step, loss))
        loss_scores.append(loss)

    time_taken = time.time() - start_time
    print("Time taken for this epoch:", time_taken)

    return 0
