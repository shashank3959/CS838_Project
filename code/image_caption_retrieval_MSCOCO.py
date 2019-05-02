import torch
from torchvision import transforms
import torch.utils.data as data
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from models import *
from data_loader import get_loader

transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
								 
def caption_list_gen(caption,batch_size):
    caption_list=[]
    for i in range(len(caption[0])):
        new_caption=[]
        for j in range(len(caption)):
            new_caption.append(caption[j][i])
        caption_list.append(new_caption)
    return caption_list


def load_model(model_path='npairs_loss_model.tar',map_location='cpu'):
    image_model = VGG19(pretrained=True)
    caption_model = LSTMBranch()
    checkpoint = torch.load(model_path, map_location)
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    image_model.load_state_dict(checkpoint['image_model'])
    caption_model.load_state_dict(checkpoint['caption_model'])
    print ("Loaded checkpoint '{}' (epoch {})".format(model_path,checkpoint['epoch']))
    return image_model, caption_model
	

def gen_matchmap(image_model, caption_model, image_tensor, caption_tensor):
    image_op = image_model(image_tensor)
    caption_op = caption_model(caption_tensor)

    n_imgs = image_op.size(0)
    matchmap_list = []

    for i in range(n_imgs):
        matchmap = matchmap_generate(image_op[i],caption_op[i])
        mm = matchmap.detach().numpy()
        matchmap_list.append(mm)

    return matchmap_list

def get_data(batch_size, fetch_mode='retrieval'):
    data_loader_val = get_loader(transform=transform,
                                  mode='val',
                                  batch_size=1,
                                  vocab_from_file=True,
                                  fetch_mode='retrieval',
                                  data_mode='imagecaption',
                                  disp_mode='imgcapretrieval',
                                  num_workers=0,
                                  test_size=100)
    dataloader_iterator = iter(data_loader_val)
	for i in range(1):
	    allimagestensor,allgtcaptions,total_caption_gloves,finalcaptions5k,img_cap_dict,img_cap_len_dict,img_imgid_dict = next(dataloader_iterator)
        break
		
	print ("Sanity Check ---> Size of the selected test Image Tensor",allimagestensor.shape)
    print ("Sanity Check ---> Size of the selected test Glove-Caption Tensor",total_caption_gloves.shape)
    
    return allimagestensor,allgtcaptions,total_caption_gloves,finalcaptions5k,img_cap_dict,img_cap_len_dict,img_imgid_dict

def get_output(image_tensor,image_model,caption_glove_tensor,caption_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_model = image_model.to(device)
    caption_model = caption_model.to(device)
    image_model.eval()
    caption_model.eval()
    image_ip_val = image_tensor.to(device)
    all_caption_output_val=[]
    caption_glove_ip_val = caption_glove_tensor.to(device)
    with torch.no_grad():
        caption_output_val = caption_model(caption_glove_ip_val)
        image_output_val = image_model(image_ip_val)
    print ("Sanity Check ---> Size of the output tensor from LSTM ",caption_output_val.shape)
    print ("Sanity Check ---> Size of the output tensor from Image Model ",image_output_val.shape)
    return image_output_val, caption_output_val
	
	

def get_sim_mat(image_output,caption_glove_tensor_output,score_type):
    sim_mat=[]
    for captiontensor in caption_glove_tensor_output:
        captiontensor = torch.unsqueeze(captiontensor,dim=0)
        sim_val = compute_matchmap_similarity_matrix(image_output,captiontensor,score_type)
        sim_mat.append(sim_val)
    return sim_mat
	

image_model, caption_model = load_model()
batch_size = 64
score_type = 'Avg_Both'
allimagestensor,allgtcaptions,total_caption_gloves,finalcaptions5k,img_cap_dict,img_cap_len_dict,img_imgid_dict = get_data(batch_size)

new_caption_glove = torch.squeeze(total_caption_gloves,dim=0)
new_image_tensor= torch.squeeze(allimagestensor,dim=0)

image_output, caption_glove_tensor_output = get_output(new_image_tensor,image_model,new_caption_glove,caption_model)


img_cap_corr = dict()
total_count = 0
for key in img_cap_len_dict.keys():
    img_cap_corr[key] = list(range(img_cap_len_dict[key]))
    img_cap_corr[key] = [item + total_count for item in img_cap_corr[key]]
    total_count += img_cap_len_dict[key].item()
   

cap_img_corr = dict( (v,k) for k in img_cap_corr for v in img_cap_corr[k] )

img_cap_corr = list(img_cap_corr.values())


sim_mat = compute_matchmap_similarity_matrix(image_output,caption_glove_tensor_output,score_type)
C2I_scores, C2I_ind = sim_mat.topk(5,0)
I2C_scores, I2C_ind = sim_mat.topk(5,1)


recall_values=calc_recalls_uneven(image_output,caption_glove_tensor_output,score_type,img_cap_corr,cap_img_corr)


print ("**********************************")
print ("The Recall Scores are:")
print (recall_values)
print ("**********************************")

sim_mat = compute_matchmap_similarity_matrix(image_output,caption_glove_tensor_output,score_type)
C2I_scores, C2I_ind = sim_mat.topk(5,0)
I2C_scores, I2C_ind = sim_mat.topk(5,1)								  