import time
import torch.utils.data as data
from utils import matchmap_generate, compute_similarity_score
import torch

def train(data_loader_train, image_model, caption_model, epoch, total_train_step, batch_size, start_step=1, start_loss=0.0):
    """Train model for exactly one epoch using the parameters given"""

    image_model.train()
    caption_model.train()

    total_loss = start_loss

    start_time = time.time()

    #for i_step in range(start_step, total_step + 1):
    for i_step in range(1, 30):
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
        if torch.cuda.is_available():
            image_ip = image_ip.cuda()
            caption_glove_ip = caption_glove_ip.cuda()

        image_output = image_model(image_ip)
        caption_glove_output = caption_model(caption_glove_ip)

        sim_scores = list()
        for sample in range(batch_size):
            mmap = matchmap_generate(image_output[sample], caption_glove_output[sample])
            score = compute_similarity_score(mmap, "Max_Img")
            sim_scores.append(score)

        #print(sim_scores)

    time_taken = time.time() - start_time
    print("Time taken for this epoch:", time_taken)

    return 0
