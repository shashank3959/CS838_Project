"""Create the CoCoDataset and a DataLoader for it."""
import nltk
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json


def get_loader(transform,
               mode="train",
               batch_size=1,
               vocab_threshold=None,
               vocab_file="./vocab.pkl",
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0,
               cocoapi_loc=".",
               vocab_glove_file="../data/vocab_glove.json",
               fetch_mode='default',
               data_mode='default'):
    """Return the data loader.
    Parameters:
        transform: Image transform.
        mode: One of "train", "val" or "test".
        batch_size: Batch size (if in testing mode, must have batch_size=1).
        vocab_threshold: Minimum word count threshold.
        vocab_file: File containing the vocabulary. 
        start_word: Special word denoting sentence start.
        end_word: Special word denoting sentence end.
        unk_word: Special word denoting unknown words.
        vocab_from_file: If False, create vocab from scratch & override any 
                         existing vocab_file. If True, load vocab from from
                         existing vocab_file, if it exists.
        num_workers: Number of subprocesses to use for data loading 
        cocoapi_loc: The location of the folder containing the COCO API: 
                     https://github.com/cocodataset/cocoapi
        vocab_glove_file: This JSON file contains the Glove embeddings for each
                    word in the vocabulary.
        fetch_mode: Indicates mode of retrieving data
        data_mode: Indicates the type of Data being retrieved for Image Retrieval Task. Can be image or text
    """
    
    assert mode in ["train", "val", "test"], "mode must be one of 'train', 'val' or 'test'."
    if vocab_from_file == False: 
        assert mode == "train", "To generate vocab from captions file, \
               must be in training mode (mode='train')."

    # Based on mode (train, val, test), obtain img_folder and annotations_file
    if mode == "train":
        if vocab_from_file == True: 
            assert os.path.exists(vocab_file), "vocab_file does not exist.  \
                   Change vocab_from_file to False to create vocab_file."
        img_folder = os.path.join(cocoapi_loc, "../data/images/train2014/")
        annotations_file = os.path.join(cocoapi_loc, "../data/annotations/captions_train2014.json")
    if mode == "val":
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file == True, "Change vocab_from_file to True."
        img_folder = os.path.join(cocoapi_loc, "../data/images/val2014/")
        annotations_file = os.path.join(cocoapi_loc, "../data/annotations/captions_val2014.json")
    if mode == "test":
        assert batch_size == 1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file == True, "Change vocab_from_file to True."
        img_folder = os.path.join(cocoapi_loc, "../data/images/test2014/")
        annotations_file = os.path.join(cocoapi_loc, "../data/annotations/image_info_test2014.json")

    # COCO caption dataset
    dataset = CoCoDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder,
                          vocab_glove_file=vocab_glove_file,
                          fetch_mode=fetch_mode,
                          data_mode=data_mode)

    if mode == "train":
        # Randomly sample a caption length, and sample indices with that length.
        [imgindices,capindices] = dataset.get_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        # data loader for COCO dataset.
        data_loader = data.DataLoader(dataset=dataset, 
                                      num_workers=num_workers,
                                      batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                              batch_size=dataset.batch_size,
                                                                              drop_last=False))
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=dataset.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)

    return data_loader

class CoCoDataset(data.Dataset):
    
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word, 
        end_word, unk_word, annotations_file, vocab_from_file, img_folder, vocab_glove_file, fetch_mode='default',data_mode='default'):
        self.data_mode = data_mode
        self.fetch_mode = fetch_mode
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder
        if self.mode == "train" or self.mode == "val":
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            self.imgIds = self.coco.getImgIds()
            print ("The total Image IDs in Val Set are:",len(self.imgIds))
            rng = random.Random(9003)
            self.imgIds1k = rng.sample(self.imgIds,1000)
            #random.seed(9002)
            #self.imgIds1k=random.sample(self.imgIds,1000)
            print ("The toal 1k Images are:",len(self.imgIds1k))
            self.capIds5k=self.coco.getAnnIds(imgIds=self.imgIds1k)
            print ("The total 5k Captions are:",len(self.capIds5k))
            #all_tokens = [nltk.tokenize.word_tokenize(
            #              str(self.coco.anns[self.ids[index]]["caption"]).lower())
            #                for index in tqdm(np.arange(len(self.ids)))]
            #self.caption_lengths = [len(token) for token in all_tokens]
        # If in test mode
        else:
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item["file_name"] for item in test_info["images"]]
        self.vocab_glove = json.load(open(vocab_glove_file, "r"))

    def __getitem__(self,index):
        # Obtain image and caption if in training or validation mode
        if self.mode == "train" or self.mode == "val" and self.data_mode=="image":
            index=self.imgIds1k
            img123=random.sample(list(np.arange(len(index))),1)[0]
            print("ASDADAD Index sampled is:", img123)
            path = self.coco.loadImgs(index)[img123]["file_name"]
            imageid = self.coco.loadImgs(index)[img123]["id"]
            ground_truth_annid=self.coco.getAnnIds(imgIds=imageid)
            ground_truth_cap = self.coco.loadAnns(ground_truth_annid)
            ground_truth_cap = [capt['caption'] for capt in ground_truth_cap]
            #ground_truth_cap = [capt['caption'] for capt in ground_truth_cap]
            print ("######################## GT ##########################")
            print(ground_truth_cap)
            image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            image = self.transform(image)
            print ("******************** AFTER TRANSFORM ****************",image.shape,path,img123)
            # For each word in caption, return its glove-representation
            if self.fetch_mode == 'default':
                # Return pre-processed image and caption tensors
                return image
            elif self.fetch_mode == 'retrieval':
                #return image, caption_gloves, caption
                return image
        elif self.mode == "train" or self.mode == "val" and self.data_mode=="caption":
            all_captions_glove=[]
            captions5k = self.coco.loadAnns(self.capIds5k)
            captions5k = [capt['caption'] for capt in captions5k]
            caption=list()
            for sent in captions5k:
                #print ("CURRENT SENTENCE IS:",sent)
                tokens = nltk.tokenize.word_tokenize(str(sent).lower())
                #print ("TOKENS",len(tokens))
                caption=list()
                caption.append(self.vocab.start_word)
                caption.extend(tokens)
                caption.append(self.vocab.end_word)
                #print ("CHECK CAPTION SHAPE")
                #print (len(caption))
                caption_gloves = torch.Tensor([self.vocab_glove[word] if word in self.vocab_glove.keys() else
                                   self.vocab_glove["<unk>"] for word in caption])
                #print ("GLOVE CAPTION SHAPE",caption_gloves.shape)
                all_captions_glove.append(caption_gloves)
            print ("ALL CAPTION GLOVE SHAPE",len(all_captions_glove))
            if self.fetch_mode=='default':
                return caption_gloves
            elif self.fetch_mode == 'retrieval':
                return all_captions_glove,captions5k

        # Obtain image if in test mode
        else:
            path = self.paths[index]

            # Convert image to tensor and pre-process using transform
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)

            # Return original image and pre-processed image tensor
            return orig_image, image

    def get_imgindices(self):
        imgindices = self.imgIds1k
        return imgindices

    def get_capindices(self):
        capindices = self.capIds5k
        return capindices

    def __len__(self):
        if self.mode == "train" or self.mode == "val":
            return len(self.ids)
        else:
            return len(self.paths)
