# -*- coding: utf-8 -*-
#MF-MNER: Multi-models Fusion for MNER in Chinese Clinical Electronic Medical Records

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




#以下这部分是Bert运行结果，不是程序，全部数据集 epoch=40的结果 “BERT-base-chinese”
# （1）最原始的
#                 precision    recall  f1-score   support
#
#         B-BODY      0.880     0.915     0.897      3079
#     B-DISEASES      0.867     0.851     0.859      1311
#         B-DRUG      0.939     0.894     0.916       480
# B-EXAMINATIONS      0.856     0.829     0.842       345
#         B-TEST      0.831     0.741     0.784       572
#    B-TREATMENT      0.848     0.827     0.837       162
#         I-BODY      0.745     0.876     0.805      2435
#     I-DISEASES      0.884     0.834     0.858      6858
#         I-DRUG      0.969     0.899     0.933      1433
# I-EXAMINATIONS      0.865     0.858     0.861       833
#         I-TEST      0.864     0.766     0.812      1614
#    I-TREATMENT      0.957     0.907     0.932      1870
#
#      micro avg      0.872     0.857     0.864     20992
#      macro avg      0.876     0.850     0.861     20992
#   weighted avg      0.875     0.857     0.865     20992


#以下这部分是BART运行结果，不是程序，全部数据集 epoch=40的结果，“fnlp/bart-base-chinese”
#                 precision    recall  f1-score   support
#
#         B-BODY      0.885     0.914     0.899      3079
#     B-DISEASES      0.865     0.860     0.863      1311
#         B-DRUG      0.951     0.921     0.935       480
# B-EXAMINATIONS      0.849     0.881     0.865       345
#         B-TEST      0.872     0.736     0.798       572
#    B-TREATMENT      0.885     0.852     0.868       162
#         I-BODY      0.750     0.875     0.808      2435
#     I-DISEASES      0.888     0.850     0.869      6858
#         I-DRUG      0.965     0.927     0.946      1433
# I-EXAMINATIONS      0.843     0.918     0.879       833
#         I-TEST      0.900     0.740     0.812      1614
#    I-TREATMENT      0.967     0.927     0.947      1870
#
#      micro avg      0.878     0.869     0.873     20992
#      macro avg      0.885     0.867     0.874     20992
#   weighted avg      0.882     0.869     0.874     20992


#以下这部分是BART运行结果，不是程序，epoch=40, 0.2训练数据集的的结果，“fnlp/bart-base-chinese”

#          precision    recall  f1-score   support
#
#         B-BODY      0.883     0.893     0.888      3079
#     B-DISEASES      0.827     0.883     0.854      1311
#         B-DRUG      0.911     0.835     0.872       480
# B-EXAMINATIONS      0.847     0.786     0.815       345
#         B-TEST      0.765     0.701     0.732       572
#    B-TREATMENT      0.862     0.809     0.834       162
#         I-BODY      0.749     0.841     0.792      2435
#     I-DISEASES      0.871     0.867     0.869      6858
#         I-DRUG      0.930     0.851     0.888      1433
# I-EXAMINATIONS      0.847     0.858     0.853       833
#         I-TEST      0.813     0.726     0.767      1614
#    I-TREATMENT      0.942     0.905     0.923      1870
#
#      micro avg      0.856     0.853     0.855     20992
#      macro avg      0.854     0.829     0.841     20992
#   weighted avg      0.858     0.853     0.855     20992
#

#以下这部分是BART运行结果，不是程序，epoch=40, 0.4训练数据集的的结果，“fnlp/bart-base-chinese”
#  precision    recall  f1-score   support
#
#         B-BODY      0.875     0.899     0.887      3079
#     B-DISEASES      0.842     0.846     0.844      1311
#         B-DRUG      0.939     0.860     0.898       480
# B-EXAMINATIONS      0.807     0.786     0.796       345
#         B-TEST      0.857     0.694     0.767       572
#    B-TREATMENT      0.836     0.821     0.829       162
#         I-BODY      0.735     0.853     0.790      2435
#     I-DISEASES      0.860     0.847     0.853      6858
#         I-DRUG      0.953     0.888     0.919      1433
# I-EXAMINATIONS      0.826     0.839     0.833       833
#         I-TEST      0.889     0.702     0.785      1614
#    I-TREATMENT      0.954     0.863     0.906      1870
#
#      micro avg      0.859     0.843     0.851     20992
#      macro avg      0.864     0.825     0.842     20992
#   weighted avg      0.863     0.843     0.851     20992


#以下这部分是BART运行结果，不是程序，epoch=40, 0.6训练数据集的的结果，“fnlp/bart-base-chinese”
               # precision    recall  f1-score   support
#
#         B-BODY      0.884     0.911     0.897      3079
#     B-DISEASES      0.862     0.871     0.866      1311
#         B-DRUG      0.938     0.915     0.926       480
# B-EXAMINATIONS      0.850     0.852     0.851       345
#         B-TEST      0.844     0.764     0.802       572
#    B-TREATMENT      0.886     0.864     0.875       162
#         I-BODY      0.754     0.869     0.807      2435
#     I-DISEASES      0.891     0.854     0.872      6858
#         I-DRUG      0.952     0.921     0.937      1433
# I-EXAMINATIONS      0.863     0.882     0.872       833
#         I-TEST      0.885     0.740     0.806      1614
#    I-TREATMENT      0.957     0.930     0.943      1870
#
#      micro avg      0.877     0.868     0.872     20992
#      macro avg      0.880     0.865     0.871     20992
#   weighted avg      0.880     0.868     0.873     20992


# 以下这部分是BART运行结果，不是程序，epoch=40, 0.8训练数据集的的结果，“fnlp/bart-base-chinese”
#                 precision    recall  f1-score   support
#
#         B-BODY      0.881     0.909     0.895      3079
#     B-DISEASES      0.860     0.874     0.867      1311
#         B-DRUG      0.936     0.919     0.927       480
# B-EXAMINATIONS      0.875     0.852     0.863       345
#         B-TEST      0.841     0.785     0.812       572
#    B-TREATMENT      0.901     0.840     0.869       162
#         I-BODY      0.756     0.867     0.808      2435
#     I-DISEASES      0.886     0.846     0.866      6858
#         I-DRUG      0.950     0.923     0.936      1433
# I-EXAMINATIONS      0.840     0.897     0.868       833
#         I-TEST      0.892     0.774     0.829      1614
#    I-TREATMENT      0.965     0.931     0.948      1870
#
#      micro avg      0.876     0.869     0.872     20992
#      macro avg      0.882     0.868     0.874     20992
#   weighted avg      0.879     0.869     0.873     20992


# 这是全部数据集，Bart带优化，bart-base-chinese-Fri Sep 22 11-12-03 2023模型运行的结果，重新运行了一遍
# 比论文里写的时候那个模型稍微有点区别，但总体差不多，就不改了
#                 precision    recall  f1-score   support
#
#         B-BODY      0.886     0.916     0.900      3079
#     B-DISEASES      0.869     0.860     0.864      1311
#         B-DRUG      0.939     0.925     0.932       480
# B-EXAMINATIONS      0.858     0.843     0.851       345
#         B-TEST      0.845     0.781     0.812       572
#    B-TREATMENT      0.921     0.858     0.888       162
#         I-BODY      0.763     0.869     0.812      2435
#     I-DISEASES      0.888     0.857     0.872      6858
#         I-DRUG      0.964     0.923     0.943      1433
# I-EXAMINATIONS      0.849     0.880     0.864       833
#         I-TEST      0.879     0.761     0.816      1614
#    I-TREATMENT      0.980     0.928     0.953      1870
#
#      micro avg      0.880     0.871     0.875     20992
#      macro avg      0.887     0.867     0.876     20992
#   weighted avg      0.883     0.871     0.876     20992

#以下这部分是Bert运行结果，带adamW优化后的结果，
# 训好的模型是这个bert-base-chinese-Fri Sep 22 12-44-54 2023不是程序，全部数据集 epoch=40的结果 “BERT-base-chinese”
#                 precision    recall  f1-score   support
#
#         B-BODY      0.880     0.910     0.895      3079
#     B-DISEASES      0.857     0.851     0.854      1311
#         B-DRUG      0.942     0.879     0.909       480
# B-EXAMINATIONS      0.848     0.826     0.837       345
#         B-TEST      0.849     0.689     0.761       572
#    B-TREATMENT      0.878     0.802     0.839       162
#         I-BODY      0.753     0.860     0.803      2435
#     I-DISEASES      0.888     0.830     0.858      6858
#         I-DRUG      0.974     0.888     0.929      1433
# I-EXAMINATIONS      0.837     0.872     0.854       833
#         I-TEST      0.881     0.691     0.774      1614
#    I-TREATMENT      0.975     0.903     0.937      1870
#
#      micro avg      0.876     0.845     0.860     20992
#      macro avg      0.880     0.833     0.854     20992
#   weighted avg      0.880     0.845     0.860     20992