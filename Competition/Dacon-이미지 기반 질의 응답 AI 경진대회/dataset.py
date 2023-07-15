import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models # 이미지
from torchvision import transforms
from PIL import Image

from transformers import GPT2Tokenizer, GPT2Model # 텍스트
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

skf = StratifiedKFold(n_splits=5)
skf_dict = dict()

df = pd.read_csv('/opt/ml/Dacon/data/train.csv')
X = df['ID'].values
y = df['answer'].values

for i, (train_index, val_index) in enumerate(skf.split(X, y)):
    skf_dict[i] = (train_index, val_index)


class VQADataset(Dataset):
    def __init__(self, df, tokenizer, transform, img_path, fold=4, train=True, is_test=False):

        self.tokenizer = tokenizer
        self.transform = transform
        self.img_path = img_path
        self.is_test = is_test
        self.fold = fold
        
        if is_test:
            self.df = df
        else:
            X = df['ID'].values
            y = df['answer'].values
            train_idx, val_idx = skf_dict[fold]
            
            if train:
                self.df = df.iloc[train_index]
            else:
                self.df = df.iloc[val_index]
        
                
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = os.path.join(self.img_path, row['image_id'] + '.jpg') # 이미지
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)

        question = row['question'] # 질문
        question = self.tokenizer.encode_plus(
            question,
            truncation=True,
            add_special_tokens=True,
            max_length=20,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        if not self.is_test:
            answer = row['answer'] # 답변
            answer = self.tokenizer.encode_plus(
                answer,
                max_length=20,
                padding='max_length',
                truncation=True,
                return_tensors='pt')
            return {
                'image': image.squeeze(),
                'question': question['input_ids'].squeeze(),
                'answer': answer['input_ids'].squeeze()
            }
        else:
            return {
                'image': image,
                'question': question['input_ids'].squeeze(),
            }