import time
import torch.utils.data as data
from utils import *
import torch
from statistics import mean
from tensorboardX import SummaryWriter

writer = SummaryWriter('../logs')


def train(data_loader_train, data_loader_val, image_model, caption_model,
          optimizer, epoch, score_type, sampler, margin, total_train_step,
          batch_size, use_gpu=False, start_step=1, start_loss=0.0):
    # Trains model for 1 Epoch
    losses = AverageMeter()
    total_loss = start_loss

    start_time = time.time()

    loss_scores = list()
    total_steps = 100
    # for i_step in range(start_step, total_train_step + 1):
    for i_step in range(start_step, total_steps+1):
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
            # mmap = matchmap_generate(image_output[sample], caption_glove_output[sample])
            # score = compute_similarity_score(mmap, "Max_Img")
            score = score_function(image_output[sample], caption_glove_output[sample], score_type)
            sim_scores.append(score)

        loss = custom_loss(image_output, caption_glove_output,
                           score_type, margin, sampler)
        loss_scores.append(loss)

        optimizer.zero_grad()
        total_loss += loss
        loss.backward()
        optimizer.step()

        losses.update(loss.data[0], image_ip.size(0))
        niter = epoch * total_steps + i_step
        writer.add_scalar('data/training_loss', losses.val, niter)

        print("Step: %d, current loss: %0.4f, avg_loss: %0.4f" % (i_step, loss, total_loss / i_step))

    time_taken = time.time() - start_time
    # print("Time taken for this epoch:", time_taken)

    return total_loss / i_step


def validate(caption_model, image_model, data_loader_val, epoch,
             score_type, sampler, margin, use_gpu):
    val_losses = AverageMeter()
    total_loss_val = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_model = image_model.to(device)
    caption_model = caption_model.to(device)
    image_model = image_model.to(device)
    image_model.eval()
    caption_model.eval()

    print('---------------------------------------------------------')

    # Lists to store recall scores
    C_r10 = []
    I_r10 = []
    C_r5 = []
    I_r5 = []
    C_r1 = []
    I_r1 = []

    total_val_steps = 10
    for i_step_val in range(1, total_val_steps+1):
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

            loss = custom_loss(image_output_val, caption_output_val,
                               score_type, margin, sampler)
            loss_scores.append(loss)
            total_loss_val += loss

        I_embeddings = []
        C_embeddings = []
        I_embeddings.append(image_output_val)
        C_embeddings.append(caption_output_val)

        image_output = torch.cat(I_embeddings)
        caption_output = torch.cat(C_embeddings)

        # Calculating recall scores
        recalls = calc_recalls(image_output, caption_output, score_type)
        C_r10.append(recalls['C_r10'])
        I_r10.append(recalls['I_r10'])
        C_r5.append(recalls['C_r5'])
        I_r5.append(recalls['I_r5'])
        C_r1.append(recalls['C_r1'])
        I_r1.append(recalls['I_r1'])

        print("Step: %d, current loss: %0.4f, avg_loss: %0.4f" % (i_step_val, loss, total_loss_val / i_step_val))

        val_losses.update(loss.data[0], image_ip_val.size(0))
        niter = epoch * total_val_steps + i_step_val
        writer.add_scalar('data/val_loss', val_losses.val, niter)
        writer.add_scalar('data/caption_R10', mean(C_r10), niter)
        writer.add_scalar('data/caption_R5', mean(C_r5), niter)
        writer.add_scalar('data/caption_R1', mean(C_r1), niter)
        writer.add_scalar('data/image_R10', mean(I_r10), niter)
        writer.add_scalar('data/image_R5', mean(I_r5), niter)
        writer.add_scalar('data/image_R1', mean(I_r1), niter)

    print(' Caption Mean R@10 {C_r10:.3f} Image Mean R@10 {I_r10:.3f}'
          .format(C_r10=mean(C_r10), I_r10=mean(I_r10)), flush=True)
    print(' Caption Mean R@5 {C_r5:.3f} Image Mean R@5 {I_r5:.3f}'
          .format(C_r5=mean(C_r5), I_r5=mean(I_r5)), flush=True)
    print(' Caption Mean R@1 {C_r1:.3f} Image Mean R@1 {I_r1:.3f}'
          .format(C_r1=mean(C_r1), I_r1=mean(I_r1)), flush=True)
    print('---------------------------------------------------------')

    return total_loss_val / i_step_val
