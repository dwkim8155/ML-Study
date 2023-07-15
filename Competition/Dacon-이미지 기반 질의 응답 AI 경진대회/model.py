import sys
sys.path.append('/opt/ml/Dacon/unilm/beit3/')

import torch
import torch.nn as nn
from modeling_finetune import beit3_large_patch16_224_vqav2

model_dict = {224: beit3_large_patch16_224_vqav2} 

class BEiT_VQA(nn.Module):
    def __init__(self,img_size):
        super().__init__()
        self.model = model_dict[img_size]()

    def forward(self, img, text, text_mask=None):
        
