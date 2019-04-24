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
from collections import defaultdict


def get_loader(transform,
               mode="train",
               batch_size=1,
               vocab_threshold=None,
               vocab_file="./vocab.pkl",
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=4,
               cocoapi_loc=".",
               vocab_glove_file="../data/vocab_glove.json",
               fetch_mode='default',
               data_mode='default',
               disp_mode='default',
               test_size=1000,
               pad_caption=True):
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
                          data_mode=data_mode,
                          disp_mode=disp_mode,
                          test_size=1000,
                          pad_caption=True)

    if mode == "train":
        # Randomly sample a caption length, and sample indices with that length.
        indices = dataset.get_indices()
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
                 end_word, unk_word, annotations_file, vocab_from_file, img_folder, vocab_glove_file,
                 fetch_mode='default',
                 pad_caption=True, pad_limit=20, data_mode='default',disp_mode='default',test_size=1000):
        self.test_size=test_size
        self.disp_mode=disp_mode
        self.data_mode=data_mode
        self.fetch_mode = fetch_mode
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
                                end_word, unk_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder
        self.pad_caption = pad_caption
        self.pad_limit = pad_limit
        
        if (self.mode == "train" or self.mode == "val") and self.disp_mode=='default':
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print("Obtaining caption lengths...")
            all_tokens = [nltk.tokenize.word_tokenize(
                str(self.coco.anns[self.ids[index]]["caption"]).lower())
                for index in tqdm(np.arange(len(self.ids)))]
            self.caption_lengths = [len(token) for token in all_tokens]
            
        elif self.mode == "val" and self.disp_mode=="imgcapretrieval":
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            self.imgIds = self.coco.getImgIds()
            rng = random.Random(9003)
            self.imgIds1k = rng.sample(self.imgIds,self.test_size)
            self.capIds5k=self.coco.getAnnIds(imgIds=self.imgIds1k)
     
        # If in test mode
        else:
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item["file_name"] for item in test_info["images"]]
        self.vocab_glove = json.load(open(vocab_glove_file, "r"))
    

    def __getitem__(self, index):
        # Obtain image and caption if in training or validation mode
        if self.mode == "train" or self.mode == "val" and self.disp_mode == "default":
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]["caption"]
            img_id = self.coco.anns[ann_id]["image_id"]
            path = self.coco.loadImgs(img_id)[0]["file_name"]

            # Convert image to tensor and pre-process using transform
            image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            image = self.transform(image)

            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = list()
            caption.append(self.vocab.start_word)
            caption.extend(tokens)
            caption.append(self.vocab.end_word)

            if self.pad_caption:
                caption.extend([self.vocab.end_word] * (self.pad_limit - len(tokens)))

            caption_gloves = torch.Tensor([self.vocab_glove[word] if word in self.vocab_glove.keys() else
                                           self.vocab_glove["<unk>"] for word in caption])
            # For each word in caption, return its glove-representation
            if self.fetch_mode == 'default':
                # Return pre-processed image and caption tensors
                return image, caption_gloves
            elif self.fetch_mode == 'retrieval':
                return image, caption_gloves, caption
                       
        if self.mode == "val" and self.disp_mode=="imgcapretrieval" and self.data_mode=="imagecaption":
            index=self.imgIds1k
            img_random=random.sample(list(np.arange(len(index))),self.batch_size)[0]
            path = self.coco.loadImgs(index)[img_random]["file_name"]
            imageid = self.coco.loadImgs(index)[img_random]["id"]
            ground_truth_annid=self.coco.getAnnIds(imgIds=imageid)
            ground_truth_cap = self.coco.loadAnns(ground_truth_annid)
            ground_truth_cap = [capt['caption'] for capt in ground_truth_cap]
            image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            image = self.transform(image)
            
            all_captions_glove=[]
            captions5k = self.coco.loadAnns(self.capIds5k)
            captions5k = [capt['caption'] for capt in captions5k]
            print("Obtaining lengths of all the test dataset captions..")
            all_tokens = [nltk.tokenize.word_tokenize(
                          str(sent).lower())
                            for sent in captions5k]
            caption=list()
            all_captions_glove=[]
            finalcaptions5k=[]
            for sent in captions5k:
                tokens = nltk.tokenize.word_tokenize(str(sent).lower())
                if len(tokens)<=self.pad_limit:
                    finalcaptions5k.append(sent)
                    caption=list()
                    caption.append(self.vocab.start_word)
                    caption.extend(tokens)
                    caption.append(self.vocab.end_word)
                    if self.pad_caption:
                        caption.extend([self.vocab.end_word] * (self.pad_limit - len(tokens)))
                    caption_gloves = torch.Tensor([self.vocab_glove[word] if word in self.vocab_glove.keys() else
                                                   self.vocab_glove["<unk>"] for word in caption])
                    all_captions_glove.append(caption_gloves)
            total_caption_gloves=torch.stack(all_captions_glove,dim=0)   
            return image,ground_truth_cap,total_caption_gloves,finalcaptions5k
               
        # Obtain image if in test mode
        else:
            path = self.paths[index]

            # Convert image to tensor and pre-process using transform
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)

            # Return original image and pre-processed image tensor
            return orig_image, image

    def get_indices(self):
        if self.pad_caption:
            all_indices = np.where([self.caption_lengths[i] <= \
                                    self.pad_limit for i in np.arange(len(self.caption_lengths))])[0]
        else:
            sel_length = np.random.choice(self.caption_lengths)
            all_indices = np.where([self.caption_lengths[i] == \
                                    sel_length for i in np.arange(len(self.caption_lengths))])[0]

        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

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


class Flickr30kData(data.Dataset):
    """`Flickr30k Entities <http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/>`_ Dataset.

    Args:
        img_root (string): Root directory where images are downloaded to.
        ann_file (string): Path to annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
    """

    def __init__(self, img_root, ann_file, transform,
                 start_word='<start>',
                 end_word='<end>',
                 unk_word='<unk>',
                 vocab_glove_file="../data/vocab_glove.json",
                 fetch_mode="default",
                 pad_caption = True,
                 pad_limit=20):
        self.transform = transform
        self.root = img_root
        self.ann_file = os.path.expanduser(ann_file)
        self.vocab_glove = json.load(open(vocab_glove_file, "r"))
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.fetch_mode = fetch_mode
        self.pad_caption = True
        self.pad_limit = pad_limit

        # Read annotations and store in a dict
        self.annotations = defaultdict(list)
        with open(self.ann_file, encoding = 'utf-8') as fh:
            for line in fh:
                img_id, caption = line.strip().split('\t')
                if len(caption.split(" ")) <= self.pad_limit:
                    self.annotations[img_id[:-2]].append(caption)

        self.ids = list(sorted(self.annotations.keys()))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the image

        Returns:
            tuple: Tuple (image, caption_glove).
            Caption_glove are the glove representation of each word in a
            tokenized caption which has been randomly samples from the
            different captions associated with an image.
        """
        img_id = self.ids[index]

        # Image
        filename = os.path.join(self.root, img_id)
        image = Image.open(filename).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Captions
        target = self.annotations[img_id]

        # Randomly sample one of the captions for this image
        # :-2 removes the comma and space in the end
        target = random.sample(target, 1)[0] # [:-2]
        tokens = nltk.tokenize.word_tokenize(str(target).lower())
        caption = list()
        caption.append(self.start_word)
        caption.extend(tokens)
        caption.append(self.end_word)

        if self.pad_caption:
            caption.extend([self.end_word] * (self.pad_limit - len(tokens)))

        caption_gloves = torch.Tensor([self.vocab_glove[word] if word in self.vocab_glove.keys() else
                                       self.vocab_glove["<unk>"] for word in caption])

        # For each word in caption, return its glove-representation
        if self.fetch_mode == 'default':
            # Return pre-processed image and caption tensors
            return image, caption_gloves
        elif self.fetch_mode == 'retrieval':
            return image, caption_gloves, target

    def __len__(self):
        # These are image ids
        return len(self.ids)
