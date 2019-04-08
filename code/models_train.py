import time
import torch.utils.data as data
from utils import matchmap_generate, compute_similarity_score, custom_loss, AverageMeter
import torch

from tensorboardX import SummaryWriter

writer = SummaryWriter('../logs')


def train(data_loader_train, data_loader_val, image_model, caption_model, optimizer, epoch, total_train_step,
          batch_size, use_gpu=False, start_step=1, start_loss=0.0):
    losses = AverageMeter()
    total_loss = start_loss

    start_time = time.time()

    loss_scores = list()

    # for i_step in range(start_step, total_train_step + 1):
    for i_step in range(1, 101):
        image_model.train()
        caption_model.train()

        indices = data_loader_train.dataset.get_indices()
        # Create a batch sampler to retrieve a batch with the sampled indices
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader_train.batch_sampler.sampler = new_sampler

        # Obtain the batch
        for batch in data_loader_train:
            image_ip, caption_glove_ip = batch[0], batch[1]
            break

        # Move to GPU if CUDA is available
        if torch.cuda.is_available() and use_gpu == True:
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
        loss_scores.append(loss)

        optimizer.zero_grad()
        total_loss += loss
        loss.backward()
        optimizer.step()

        losses.update(loss.data[0], image_ip.size(0))
        niter = epoch + i_step
        writer.add_scalar('data/training_loss', losses.val, niter)

        print("Step: %d, current loss: %0.4f, avg_loss: %0.4f" % (i_step, loss, total_loss / i_step))

    time_taken = time.time() - start_time
    # print("Time taken for this epoch:", time_taken)

    return total_loss / i_step


def validate(caption_model, image_model, data_loader_val, epoch, use_gpu):
    val_losses = AverageMeter()
    total_loss_val = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_model = image_model.to(device)
    caption_model = caption_model.to(device)
    image_model = image_model.to(device)
    image_model.eval()
    caption_model.eval()

    print('---------------------------------------------------------')

    for i_step_val in range(1, 11):
        indices = data_loader_val.dataset.get_indices()
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader_val.batch_sampler.sampler = new_sampler

        for batch in data_loader_val:
            image_ip_val, caption_glove_ip_val = batch[0], batch[1]
            break

        image_ip_val = image_ip_val.to(device)
        caption_glove_ip_val = caption_glove_ip_val.to(device)

        loss_scores = list()

        with torch.no_grad():
            image_output_val = image_model(image_ip_val)
            caption_output_val = caption_model(caption_glove_ip_val)

            loss = custom_loss(image_output_val, caption_output_val)
            loss_scores.append(loss)
            total_loss_val += loss

        print("Step: %d, current loss: %0.4f, avg_loss: %0.4f" % (i_step_val, loss, total_loss_val / i_step_val))

        val_losses.update(loss.data[0], image_ip_val.size(0))
        niter = epoch + i_step_val
        writer.add_scalar('data/validation_loss', val_losses.val, niter)

    print('---------------------------------------------------------')
    return total_loss_val / i_step_val