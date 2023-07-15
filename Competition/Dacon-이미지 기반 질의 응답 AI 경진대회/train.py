import sys
sys.path.append('/opt/ml/Dacon/unilm/beit3/')

import os
import pandas as pd
from tqdm import tqdm
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import wandb
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models # 이미지
from torchvision import transforms
from torchscale.component.embedding import VisionEmbedding
from transformers import XLMRobertaTokenizer, GPT2Tokenizer
from modeling_finetune import beit3_large_patch16_224_vqav2
from dataset import VQADataset


############## PARSE ARGUMENT ########################
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--load_from", type=bool, default=True)
    parser.add_argument("--chk_path", type=str, default='/opt/ml/Dacon/weight/BEiT/BEiT_latest_model.pt')
    parser.add_argument("--pretrain", type=bool, default=True)
    
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr",  type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=int, default=0.00001)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--amp", type=bool, default=True)           ## Use Auto mixed precision or not
    # parser.add_argument("--accum", type=int, default=1)             ## gradient accumulation (BATCH = batch_size * accum)

    args = parser.parse_args()
    return args


############## TRAINING SETTINGS ########################
args = parse_args()

device = 'cuda'if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'BEiT'
LOAD_FROM = args.load_from
CHK_PATH = args.chk_path
PRETRAIN = args.pretrain

#data
FOLD = args.fold
BATCH_SIZE = args.batch_size

#train 
NUM_EPOCHS = args.epochs
LR = args.lr
REG = args.weight_decay
VAL_EVERY = args.val_interval
AMP = args.amp

SAVED_DIR = f'/opt/ml/Dacon/weight/{MODEL_NAME}'


############## DATASET SETTINGS ######################
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')
sample_submission = pd.read_csv('../data/sample_submission.csv')
train_img_path = '../data/image/train'
test_img_path = '../data/image/test'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
vocab_size = len(tokenizer)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = VQADataset(train_df, tokenizer, transform, train_img_path, fold=FOLD,  train=True, is_test=False)
train_loader = DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = VQADataset(train_df, tokenizer, transform, train_img_path, fold=FOLD,  train=False, is_test=False)
val_loader = DataLoader(val_dataset, num_workers=2, batch_size=BATCH_SIZE, shuffle=True)

assert len(train_dataset) + len(val_dataset) == len(train_df)



def train(model, train_loader, val_loader, optimizer, criterion, AMP, SAVED_DIR):
    run = wandb.init(project="Dacon-VQA", entity="oif", name=MODEL_NAME, resume=False)
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    best_acc = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f'\nEPOCH{epoch+1} starts\n')
        train_loss = 0
        train_acc = 0
    
        for step, data in enumerate(train_loader):
            images = data['image'].to(device, non_blocking=True)
            question = data['question'].to(device, non_blocking=True)
            answer = data['answer'].to(device, non_blocking=True)
    
            optimizer.zero_grad(set_to_none=True)        
    
            with torch.cuda.amp.autocast(enabled=AMP):
                outputs = model(images, question, None)
                loss = criterion(outputs.view(-1, outputs.size(-1)), answer.view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            #acc
            max_index = torch.argmax(outputs, axis=-1)
            out_eq = (max_index == answer)
            for o in out_eq:
                if all(o):
                    train_acc+=1
            
            if (step + 1) % 300 == 0:
                print(
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
                
        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)
        wandb.log({"Train Loss": train_loss, "Train Acc":train_acc})
        
        if (epoch + 1) % VAL_EVERY == 0:
            val_loss = 0
            val_acc = 0
            with torch.no_grad():
                model.eval()
                print("\nEvaluating validation results...")
                for data in val_loader:
                    images = data['image'].to(device, non_blocking=True)
                    question = data['question'].to(device, non_blocking=True)
                    answer = data['answer'].to(device, non_blocking=True)
            
                    outputs = model(images, question, None)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), answer.view(-1))

                    val_loss += loss.item()
                    
                    #acc
                    max_index = torch.argmax(outputs, axis=-1)
                    out_eq = (max_index == answer)
                    for o in out_eq:
                        if all(o):
                            val_acc+=1
                
            val_loss /= len(val_loader)
            val_acc /= len(val_loader.dataset)
            wandb.log({"Valid Loss": val_loss, "valid Acc":val_acc})
        
        print('(Train) Mean loss: {:.4f} | Train Acc: {:.4f}'.format(train_loss, train_acc))
        print('(Val)   Mean loss: {:.4f} | Valid ACC: {:.4f}'.format(val_loss, val_acc))
        
        
        os.makedirs(SAVED_DIR, exist_ok=True)
        output_path = os.path.join(SAVED_DIR, f'{MODEL_NAME}_latest_model.pt')
        torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                }, output_path)
        
        print(f"Latest Model Saved in {SAVED_DIR}")
        
        if best_acc < val_acc:
            print(f"Best performance at epoch: {epoch + 1}, {best_acc:.4f} -> {val_acc:.4f}")
            print(f"Save model in {SAVED_DIR}")
            best_acc = val_acc
                
            output_path = os.path.join(SAVED_DIR, f'{MODEL_NAME}_best_model.pt')
            torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                }, output_path)
    
    wandb.finish()     


if __name__ == '__main__':

    if PRETRAIN:
        model = beit3_large_patch16_224_vqav2(pretrained=True).cuda()
        chkp = '/opt/ml/Dacon/weight/pretrain/beit/beit3_large_patch16_224.pth'
        model.load_state_dict(torch.load(chkp)['model'], strict=False)
        print('pretrained weights loaded!!')
    else:
        model = beit3_large_patch16_224_vqav2(pretrained=True).cuda()

    if LOAD_FROM:
        model.load_state_dict(torch.load(CHK_PATH)['model_state_dict'])
        print('loaded weights')
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(params=model.parameters(), lr=LR, weight_decay=REG)
    
    train(model, train_loader, val_loader, optimizer, criterion, AMP, SAVED_DIR)