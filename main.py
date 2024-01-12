# -*- coding: utf-8 -*-
#MF-MNER: Multi-models Fusion for MNER in Chinese Clinical Electronic Medical Records
@Author:Haoze Du1, Jiahao Xu, Zhiyong Du, Lihui Chen, Shaohui Ma, Junwei Cui, Dongqing Wei, Xianfang Wang
@Date: 2023-09-27
@LastEditTime: 2023-10-6
@Description: This file is for building model. 
@All Right Reserve

# 
#
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import os
import warnings
import argparse
import numpy as np
from sklearn import metrics
from models import Bert_BiLSTM_CRF
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import NerDataset, PadBatch, VOCAB, tag2idx, idx2tag
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from sklearn.metrics import f1_score
from tqdm import tqdm 


def train(e, model, iterator, optimizer, scheduler, device):
    model.train()
    all_loss = []
    losses = 0.0
    step = 0
    for i, batch in tqdm(enumerate(iterator)):
        step += 1
        x, y, z = batch
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)

        loss = model(x, y, z)
        losses += loss.item()
        all_loss.append(loss.item())
        """ Gradient Accumulation """
        '''
          full_loss = loss / 2                            # normalize loss 
          full_loss.backward()                            # backward and accumulate gradient
          if step % 2 == 0:             
              optimizer.step()                            # update optimizer
              scheduler.step()                            # update scheduler
              optimizer.zero_grad()                       # clear gradient
        '''
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    print("Epoch: {}, Loss:{:.4f}".format(e, losses/step))
    return losses/step

def validate(e, model, iterator, device):
    model.eval()
    Y, Y_hat = [], []
    losses = 0
    step = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            step += 1

            x, y, z = batch
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)

            y_hat = model(x, y, z, is_test=True)

            loss = model(x, y, z)
            losses += loss.item()
            # Save prediction
            for j in y_hat:
              Y_hat.extend(j)
            # Save labels
            mask = (z==1)
            y_orig = torch.masked_select(y, mask)
            Y.append(y_orig.cpu())

    Y = torch.cat(Y, dim=0).numpy()
    Y_hat = np.array(Y_hat)
    acc = (Y_hat == Y).mean()*100

    print("Epoch: {}, Val Loss:{:.4f}, Val Acc:{:.3f}%".format(e, losses/step, acc))
    return model, losses/step, acc

def test(model, iterator, device):
    model.eval()
    Y, Y_hat = [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x, y, z = batch
            x = x.to(device)
            z = z.to(device)
            y_hat = model(x, y, z, is_test=True)
            # Save prediction
            for j in y_hat:
              Y_hat.extend(j)
            # Save labels
            mask = (z==1).cpu()
            y_orig = torch.masked_select(y, mask)
            Y.append(y_orig)

    Y = torch.cat(Y, dim=0).numpy()
    y_true = [idx2tag[i] for i in Y]
    y_pred = [idx2tag[i] for i in Y_hat]


    return y_true, y_pred
import time
import pandas as pd

if __name__=="__main__":

    labels = ['B-BODY', 'B-DISEASES', 'B-DRUG', 'B-EXAMINATIONS', 'B-TEST', 'B-TREATMENT',
      'I-BODY', 'I-DISEASES', 'I-DRUG', 'I-EXAMINATIONS', 'I-TEST', 'I-TREATMENT']
    
    best_model = None
    _best_val_loss = 1e18
    _best_val_acc = 1e-18

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_name_or_path", type=str,
                        default="bert-base-chinese", 
                        # default="fnlp/bart-base-chinese",
                        )

    parser.add_argument("--output_dir", default="results/bert-base-chinese")。
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=40) 
   
    parser.add_argument("--trainset", type=str, default="./CCKS_2019_Task1/processed_data/train_dataset.txt")
  。
    parser.add_argument("--validset", type=str, default="./CCKS_2019_Task1/processed_data/val_dataset.txt")
 
    parser.add_argument("--testset", type=str, default="./CCKS_2019_Task1/processed_data/test_dataset.txt")

 
    ner = parser.parse_args()
  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

   
    model = Bert_BiLSTM_CRF(ner.model_name_or_path, tag_to_ix=tag2idx, embedding_dim=768, hidden_dim=256).cuda()

    print('Initial model Done.')
    
    train_dataset = NerDataset(ner.model_name_or_path, ner.trainset)
    
    eval_dataset = NerDataset(ner.model_name_or_path, ner.validset)
   
    test_dataset = NerDataset(ner.model_name_or_path, ner.testset)
    print('Load Data Done.')
   
   train_iter = data.DataLoader(dataset=train_dataset,
                                batch_size=ner.batch_size,
                                 shuffle=True,
                                num_workers=4,
    #                              collate_fn=PadBatch)
   
  
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                  batch_size=(ner.batch_size)//2,
                                 shuffle=False,
                                  num_workers=4,
                                 collate_fn=PadBatch)
    
   
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=(ner.batch_size)//2,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=PadBatch)

   
    import math
    optimizer = optim.Adam(model.parameters(), lr=ner.lr, weight_decay=0.01)
    optimizer = AdamW(model.parameters(), lr=ner.lr, eps=1e-6)
    #
    # Warmup
    len_dataset = len(train_dataset)
     epoch = ner.n_epochs
     batch_size = ner.batch_size
    total_steps = (len_dataset // batch_size) * epoch if len_dataset % batch_size == 0 else (len_dataset // batch_size + 1) * epoch
    #
     warm_up_ratio = 0.1 
     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)
    
     print('Start Train...,')
     train_loss, dev_loss, dev_acc = [], [], []
    
     for epoch in range(1, ner.n_epochs+1):
         tloss = train(epoch, model, train_iter, optimizer, scheduler, device)
         train_loss.append(tloss)
         candidate_model, loss, acc = validate(epoch, model, eval_iter, device)
         dev_loss.append(loss)
         dev_acc.append(acc)
    
         if loss < _best_val_loss and acc > _best_val_acc:
             best_model = candidate_model
            _best_val_loss = loss
             _best_val_acc = acc
    
    # print("=============================================")
    # #
    
   output_dir = ner.output_dir+ "-"+ "-".join(time.asctime(time.localtime()).split(":")) 
   print(" saving best checkpoints to output_dir")
   torch.save(best_model, output_dir)
   df = pd.DataFrame(
         {"train_loss": train_loss,
         "dev_loss": dev_loss,
          "dev_acc": dev_acc}
     )
     df.to_csv(output_dir + ".csv")

    
    # # results/bart-base-chinese-Sat Sep 16 17-52-06 2023
    # # results/bart-base-chinese-Thu Sep  7 15-54-11 2023
    # # results/bart-base-chinese-Tue Sep 12 15-50-44 2023
    # # results/bart-base-chinese-Tue Sep 12 17-43-07 2023
    # # results/bart-base-chinese-Tue Sep 12 18-46-20 2023
    # # results/bart-base-chinese-Wed Sep 13 09-51-40 2023

    # # results/bart-base-chinese-Sat Sep 16 17-52-06 2023

    best_model = torch.load("results/bert-base-chinese-Fri Sep 22 12-44-54 2023")

   
     print(best_model)
     print()
    for name, parameters in best_model.named_parameters():
        print(name, ':', parameters.size())

    y_test, y_pred = test(best_model, test_iter, device)
     print(y_test, y_pred)
    
    print(metrics.classification_report(y_test, y_pred, labels=labels, digits=3))
     y_scores = test.predict_proba(y_test)





