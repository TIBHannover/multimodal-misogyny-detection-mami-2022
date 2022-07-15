## Good Explanation here: https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
import torch
import numpy as np
import re
import itertools

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

from torch import nn
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, sampler
from torchvision import models, transforms

import pandas as pd
import os, random, copy, re
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CustomDatasetFixed(Dataset):
    def __init__(self, data_df, phase, img_transform, preprocess, tokenize, max_length):
        self.data_df = data_df
        self.phase = phase
        self.img_transform = img_transform
        self.preprocess = preprocess
        self.tokenize = tokenize
        self.max_len = max_length
        self.dloc = 'data/'

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        fidx = self.data_df['file_name'][idx].strip('.jpg')
        text = self.data_df['text'][idx]
        label1 = int(self.data_df['label'][idx])
        label2 = int(self.data_df['shaming'][idx])
        label3 = int(self.data_df['stereotype'][idx])
        label4 = int(self.data_df['objectification'][idx])
        label5 = int(self.data_df['violence'][idx])

        img = Image.open(os.path.join(self.dloc, self.phase+'_images', fidx+'.jpg')).convert('RGB')
        img = self.img_transform(img)

        proc_text = self.preprocess(text)
        text_tokens, masks = self.tokenize(proc_text)

        return img, text_tokens, masks, label1, label2, label3, label4, label5


class CustomDatasetAug(Dataset):
    def __init__(self, root, phase, csv_file, transform, metric, tokenizer):
        self.data_df = pd.read_csv(os.path.join(root,csv_file))
        # if metric == 'sc' and phase == 'train':
        #     self.data_df = data_df[data_df['sc'] != -1.0].reset_index(drop=True)
        # elif metric == 'sc' and phase == 'test':
        #     self.data_df = data_df[data_df['sc'] != -1].reset_index(drop=True)

        self.root = root
        self.phase = phase
        self.transform = transform
        self.metric = metric
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        text = self.data_df['text'][idx]
        path = self.data_df['path'][idx]
        label = int(self.data_df[self.metric][idx])
        if label == -1:
            label = 2

        img = Image.open(self.root+'/'+self.phase+'/'+path).convert('RGB')
        img = self.transform(img)
        enc_dict = self.tokenizer.encode_plus(
                        text,                      
                        add_special_tokens = True, 
                        padding = 'max_length',
                        max_length = 203, 
                        truncation=True,     
                        return_attention_mask = True, 
                        return_tensors = 'pt',     # Return pytorch tensors.
                   ) 

        return img, enc_dict['input_ids'], enc_dict['attention_mask'], label


def get_text_processor(word_stats='twitter', keep_hashtags=False):
    return TextPreProcessor(
            # terms that will be normalized , 'number','money', 'time','date', 'percent' removed from below list
            normalize=['url', 'email', 'phone', 'user'],
            # terms that will be annotated
            annotate={"hashtag","allcaps", "elongated", "repeated",
                      'emphasis', 'censored'},
            fix_html=True,  # fix HTML tokens

            # corpus from which the word statistics are going to be used
            # for word segmentation
            segmenter=word_stats,

            # corpus from which the word statistics are going to be used
            # for spell correction
            corrector=word_stats,

            unpack_hashtags=keep_hashtags,  # perform word segmentation on hashtags
            unpack_contractions=True,  # Unpack contractions (can't -> can not)
            spell_correct_elong=True,  # spell correction for elongated words

            # select a tokenizer. You can use SocialTokenizer, or pass your own
            # the tokenizer, should take as input a string and return a list of tokens
            tokenizer=SocialTokenizer(lowercase=True).tokenize,

            # list of dictionaries, for replacing tokens extracted from the text,
            # with other expressions. You can pass more than one dictionaries.
            dicts=[emoticons]
        )


def get_word_sent_embedding(tweet, model, tokenizer, device):
    # Split the sentence into tokens.
    input_ids = torch.tensor([tokenizer.encode(tweet, add_special_tokens=True)]).to(device)
    if input_ids.shape[1] > 128:
        last_token = input_ids[0, -1]
        input_ids = input_ids[0, :128].unsqueeze(0)
        input_ids[0, 127] = last_token
    
    # Predict hidden states features for each layer
    with torch.no_grad():
        try:
            last_out, _, encoded_layers = model(input_ids, return_dict=False)
        except:
            last_out, encoded_layers = model(input_ids, return_dict=False)


    # Last Layer word embeddings average
    sent_emb_last = torch.mean(last_out[0], dim=0).cpu().numpy()

    # Concatenate the tensors for all layers.
    # Output is [1 x 12 x |W| x 768], |W| -> Number of words
    token_embeddings = torch.stack(encoded_layers, dim=0)

    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)

    # Swap dimensions 0 and 1.
    # Output is [|W| x 12 x 768]
    token_embeddings = token_embeddings.permute(1,0,2)

    # Stores the concatenated (last 4 layers) token vectors
    token_vecs_cat = []

    # Loop over tokens
    for token in token_embeddings:
        # Concatenate the vectors (that is, append them together) from the last 
        # four layers.
        # Concatenated length becomes 3072 (768*4)
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
   
        token_vecs_cat.append(cat_vec.cpu().numpy())

    ## Average over words, Input [|w| x 3072] -> Output [3072]
    sent_word_catavg = np.mean(token_vecs_cat, axis=0)

    # Stores the sum (last 4 layers) of token vectors
    token_vecs_sum = []   

    # Loop over tokens
    for token in token_embeddings:
        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)
        
        token_vecs_sum.append(sum_vec.cpu().numpy())

    ## Average over words, Input [|w| x 768] -> Output [768]
    sent_word_sumavg = np.mean(token_vecs_sum, axis=0)

    # Second last layer tokens
    token_vecs = encoded_layers[-2][0]

    # Average of second last layer
    # Input [|w| x 768] -> Output [768]
    sent_emb_2_last = torch.mean(token_vecs, dim=0).cpu().numpy()

    return sent_word_catavg, sent_word_sumavg, sent_emb_2_last, sent_emb_last, input_ids.shape[1]-2


def get_word_sent_embedding_maxlen(tweet, model, tokenizer, device, max_len):
    # Split the sentence into tokens.
    enc_dict = tokenizer.encode_plus(tweet, 
                    add_special_tokens=True, 
                    padding = 'max_length',
                    max_length = max_len, 
                    truncation=True,     
                    return_attention_mask = True, 
                    return_tensors = 'pt')
    
    # Predict hidden states features for each layer
    with torch.no_grad():
        try:
            last_out, _, encoded_layers = model(input_ids=enc_dict['input_ids'].to(device), 
                        attention_mask=enc_dict['attention_mask'].to(device), return_dict=False)
        except:
            last_out, encoded_layers = model(input_ids=enc_dict['input_ids'].to(device), 
                        attention_mask=enc_dict['attention_mask'].to(device), return_dict=False)


    # Last Layer word embeddings average
    sent_emb_last = torch.mean(last_out[0], dim=0).cpu().numpy()

    # Concatenate the tensors for all layers.
    # Output is [1 x 12 x |W| x 768], |W| -> Number of words
    token_embeddings = torch.stack(encoded_layers, dim=0)

    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)

    # Swap dimensions 0 and 1.
    # Output is [|W| x 12 x 768]
    token_embeddings = token_embeddings.permute(1,0,2)

    # Stores the concatenated (last 4 layers) token vectors
    token_vecs_cat = []

    # Loop over tokens
    for token in token_embeddings:
        # Concatenate the vectors (that is, append them together) from the last 
        # four layers.
        # Concatenated length becomes 3072 (768*4)
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
   
        token_vecs_cat.append(cat_vec.cpu().numpy())

    ## Average over words, Input [|w| x 3072] -> Output [3072]
    sent_word_catavg = np.mean(token_vecs_cat, axis=0)

    # Stores the sum (last 4 layers) of token vectors
    token_vecs_sum = []   

    # Loop over tokens
    for token in token_embeddings:
        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)
        
        token_vecs_sum.append(sum_vec.cpu().numpy())

    ## Average over words, Input [|w| x 768] -> Output [768]
    sent_word_sumavg = np.mean(token_vecs_sum, axis=0)

    # Second last layer tokens
    token_vecs = encoded_layers[-2][0]

    # Average of second last layer
    # Input [|w| x 768] -> Output [768]
    sent_emb_2_last = torch.mean(token_vecs, dim=0).cpu().numpy()

    return sent_word_catavg, sent_word_sumavg, sent_emb_2_last, sent_emb_last, enc_dict['input_ids'].shape[1]-2
