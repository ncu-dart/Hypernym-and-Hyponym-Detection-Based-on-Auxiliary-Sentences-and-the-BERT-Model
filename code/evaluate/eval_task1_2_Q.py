#!/usr/bin/env python
# coding: utf-8
from argparse import ArgumentParser

import torch
import numpy as np
import pandas as pd
from sklearn import metrics
import math
from scipy.stats import spearmanr
import torch.nn as nn

from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertForSequenceClassification
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


parser = ArgumentParser()
parser.add_argument("-_model1_path", help="path to model", dest="model1_path", default = '../../output/task1_Q_18.pth')
parser.add_argument("-_model2_path", help="path to model", dest="model2_path", default = '../../output/task2_Q_6.pth')
args = parser.parse_args()



PRETRAINED_MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
NUM_LABELS = 2
minus_num = 20
# 讓模型跑在 GPU 上並取得訓練集的分類準確率

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
BATCH_SIZE = 64



print('Bless Hyper', '-'*minus_num)
class BlessDataset(Dataset):
# 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, tokenizer):
        self.df  = pd.read_csv(path, sep="\t", header=None)
        self.len = len(self.df)
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer

    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        text_a, text_b , label1= self.df.iloc[idx, :].values
        # 將 label 文字也轉換成索引方便轉換成 tensor

        # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
        word_pieces = ["[CLS]"]


        fake_a = ""
        tokens_a = self.tokenizer.tokenize(fake_a)
        fake_b = ""
        tokens_b = self.tokenizer.tokenize(fake_b)
        
        tail = "is a type of "
        token_tail = self.tokenizer.tokenize(tail)
        try:
            tokens_a = self.tokenizer.tokenize(text_a)
        except:
            print(idx, text_a, text_b)
            exit()
        
        try:
            tokens_b = self.tokenizer.tokenize(text_b)
        except:
            print(idx, text_a, text_b)
            exit()
        if(label1 == True):
            label1_tensor = torch.tensor(1)
        else:
            label1_tensor = torch.tensor(0)    

        word_pieces += (tokens_a + token_tail + tokens_b + ["[SEP]"])
        len_a = len(word_pieces)

 
        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a, 
                                        dtype=torch.long)


  

        return (tokens_tensor, segments_tensor, label1_tensor)
    
    def __len__(self):
        return self.len


path = "../../data/evaluate/lexical_entailment/task1/bless2011/data.tsv"
testset2 = BlessDataset(tokenizer=tokenizer)

def create_mini_batch_bless(samples):
    tokens_tensors1 = [s[0] for s in samples]
    segments_tensors1 = [s[1] for s in samples]

    
    # 測試集有 labels
    if samples[0][2] is not None:
        label1_ids = torch.stack([s[2] for s in samples])
    

    # zero pad 到同一序列長度
    tokens_tensors1 = pad_sequence(tokens_tensors1, 
                                  batch_first=True)
    segments_tensors1 = pad_sequence(segments_tensors1, 
                                    batch_first=True)
    

    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors1 = torch.zeros(tokens_tensors1.shape, 
                                dtype=torch.long)
    masks_tensors1 = masks_tensors1.masked_fill(
        tokens_tensors1 != 0, 1)

    return tokens_tensors1, segments_tensors1, masks_tensors1, label1_ids


testloader2 = DataLoader(testset2, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch_bless, shuffle = False)


bless  = pd.read_csv(path, sep="\t", header=None)

score_board_Col = ['task1_res', 'task2_res', 'task1_score', 'task2_score', 'y_pred', 'rank_score', 'task1_prob','task2_prob']
score_board = pd.DataFrame(columns = score_board_Col)

score_board.insert(0, 'word1', bless[0])
score_board.insert(1, 'word2', bless[1])
score_board.insert(8, 'y_true', bless[2])


def get_prediction_bless(model, dataloader):
    correct = 0
    fun_correct = 0
    total = 0
    score_board_index = 0
    
    with torch.no_grad():
        # 遍巡整個資料集
        for data in (dataloader):
            # 將所有 tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            
            tokens_tensors1, segments_tensors1, masks_tensors1 = data[:3]
            outputs1 = model(input_ids=tokens_tensors1, 
                            token_type_ids=segments_tensors1, 
                            attention_mask=masks_tensors1)
            
            
            Softmax = nn.Softmax(dim = 1)

            logits1= outputs1
            logits1 = Softmax(outputs1)
#             print(logits1)
            _, pred1 = torch.max(logits1.data, 1)

            # 用來計算訓練集的分類準確率
            label1s = data[3]

            for i in range(pred1.size()[0]):
                score_board.iloc[score_board_index, 3] = pred1[i].item()
                if pred1[i].item():
                    if score_board.iloc[score_board_index, 8]:
                        correct+=1
                score_board_index += 1    
    
    val_df = score_board[(score_board.y_true == True)].copy()

    total = len(val_df)
    acc = correct / total    
    
    return acc
    

model_path = args.model2_path
    
state_dict = torch.load(model_path)
model_fine = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS, state_dict=state_dict)
model_fine.eval()
model_fine = model_fine.to(device)
acc = get_prediction_bless(model_fine, testloader2)
print("Accuracy:", acc,sep = "  ")
del model_fine
torch.cuda.empty_cache()
print('')



class HeirachicalDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, tokenizer):
        self.df  = pd.read_csv(path, sep="\t", header=None)
        self.len = len(self.df)
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer

    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        text_a, text_b , label1, label2= self.df.iloc[idx, :].values
        # 將 label 文字也轉換成索引方便轉換成 tensor

        # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
        word_pieces = ["[CLS]"]


        fake_a = ""
        tokens_a = self.tokenizer.tokenize(fake_a)
        
        fake_b = ""
        tokens_b = self.tokenizer.tokenize(fake_b)
        
        middle = "and"
        token_middle = self.tokenizer.tokenize(middle)
        
        tail = "are hierarchically related"
        token_tail = self.tokenizer.tokenize(tail)

        try:
            tokens_a = self.tokenizer.tokenize(text_a)
        except:
            print(idx, text_a, text_b)
            exit()
        
        try:
            tokens_b = self.tokenizer.tokenize(text_b)
        except:
            print(idx, text_a, text_b)
            exit()
        if(label1 == True):
            label1_tensor = torch.tensor(1)
        else:
            label1_tensor = torch.tensor(0)
            
        if label2 == "hyper":
            label2_tensor = torch.tensor(1)
        elif label2 == "rhyper":
            label2_tensor = torch.tensor(-1)
        elif label2 == "other":
            label2_tensor = torch.tensor(0)
    

        
        word_pieces += (tokens_a + token_middle + tokens_b + token_tail + ["[SEP]"])
        len_a = len(word_pieces)

 
        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a, 
                                        dtype=torch.long)


  

        return (tokens_tensor, segments_tensor, label1_tensor, label2_tensor)
    
    def __len__(self):
        return self.len



class DirectionDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, tokenizer):
        self.df  = pd.read_csv(path, sep="\t", header=None)
        self.len = len(self.df)
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer

    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        text_a, text_b , label1, label2= self.df.iloc[idx, :].values
        # 將 label 文字也轉換成索引方便轉換成 tensor

        # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
        word_pieces = ["[CLS]"]


        fake_a = ""
        tokens_a = self.tokenizer.tokenize(fake_a)
        fake_b = ""
        tokens_b = self.tokenizer.tokenize(fake_b)
        
        tail = "is a type of "
        token_tail = self.tokenizer.tokenize(tail)
        try:
            tokens_a = self.tokenizer.tokenize(text_a)
        except:
            print(idx, text_a, text_b)
            exit()
        
        try:
            tokens_b = self.tokenizer.tokenize(text_b)
        except:
            print(idx, text_a, text_b)
            exit()
        if(label1 == True):
            label1_tensor = torch.tensor(1)
        else:
            label1_tensor = torch.tensor(0)
            
        if label2 == "hyper":
            label2_tensor = torch.tensor(1)
        elif label2 == "rhyper":
            label2_tensor = torch.tensor(-1)
        elif label2 == "other":
            label2_tensor = torch.tensor(0)
    

        
        word_pieces += (tokens_a + token_tail + tokens_b + ["[SEP]"])
        len_a = len(word_pieces)

 
        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a, 
                                        dtype=torch.long)

        return (tokens_tensor, segments_tensor, label1_tensor, label2_tensor)
    
    def __len__(self):
        return self.len



def create_mini_batch(samples):
    tokens_tensors1 = [s[0] for s in samples]
    segments_tensors1 = [s[1] for s in samples]

    
    # 測試集有 labels
    if samples[0][2] is not None:
        label1_ids = torch.stack([s[2] for s in samples])
    else:
        label1_ids = None
    
    if samples[0][3] is not None:
        label2_ids = torch.stack([s[3] for s in samples])

    # zero pad 到同一序列長度
    tokens_tensors1 = pad_sequence(tokens_tensors1, 
                                  batch_first=True)
    segments_tensors1 = pad_sequence(segments_tensors1, 
                                    batch_first=True)
    

    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors1 = torch.zeros(tokens_tensors1.shape, 
                                dtype=torch.long)
    masks_tensors1 = masks_tensors1.masked_fill(
        tokens_tensors1 != 0, 1)
    
    return tokens_tensors1, segments_tensors1, masks_tensors1, label1_ids, label2_ids


path = '../../data/evaluate/lexical_entailment/task2/BLESS/ABIBLESS.txt'
bibless  = pd.read_csv(path, sep="\t", header=None)

score_board_Col = ['task1_res', 'task2_res', 'task1_score', 'task2_score', 'y_pred', 'rank_score','task1_prob', 'task2_prob']
score_board = pd.DataFrame(columns = score_board_Col)

score_board.insert(0, 'word1', bibless[0])
score_board.insert(1, 'word2', bibless[1])
score_board.insert(8, 'y_true', bibless[3])

score_board.head()



def get_predictions1(model, dataloader):
    predictions = []
    single_guess = 1
    correct = 0
    fun_correct = 0
    total = 0
    y_pred = []
    y_true = []
    pred_label = []
    score_board_index = 0
    
    with torch.no_grad():
        # 遍巡整個資料集
        for data in (dataloader):
            # 將所有 tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            
            
            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors1, segments_tensors1, masks_tensors1 = data[:3]
            outputs1 = model(input_ids=tokens_tensors1, 
                            token_type_ids=segments_tensors1, 
                            attention_mask=masks_tensors1)
            
            
            Softmax = nn.Softmax(dim = 1)

            logits1= outputs1
            logits1 = Softmax(outputs1)
            _, pred1 = torch.max(logits1.data, 1)

            # 用來計算訓練集的分類準確率
            label1s = data[3]
            label2s = data[4]

            for i in range(pred1.size()[0]):
                y_true.append(label1s[i].item())
                y_pred.append(logits1[i][1].item())
                pred_label.append(pred1[i].item())
                score_board.iloc[score_board_index, 2] = pred1[i].item()
                score_board.iloc[score_board_index, 9] = logits1[i][1].item()
                score_board.iloc[score_board_index, 4] = math.log(logits1[i][1].item() / logits1[i][0].item())
                score_board_index += 1
#        
               
    d = {'y_true': y_true, 'pred_label': pred_label, 'y_pred': y_pred}
    df = pd.DataFrame(data=d)
    df.sort_values(by = 'y_pred', ascending=False, inplace=True)

    correct = (df['y_true'] == df['pred_label']).sum()
    
    
    f1 = metrics.f1_score(df["y_true"], df["pred_label"])
    
    fpr, tpr, threshold = metrics.roc_curve(df["y_true"], df["y_pred"])
    roc_auc = metrics.auc(fpr, tpr)
    print('task1 ROC AUC:', roc_auc)
    
    return predictions, acc, roc_auc, correct, total, f1
    



def get_predictions2(model, dataloader):
    predictions = []
    single_guess = 1
    correct = 0
    fun_correct = 0
    total = 0
    y_pred = []
    y_true = []
    pred_label = []
    score_board_index = 0
    shift_num = abs(math.log(1e-7))
    
    with torch.no_grad():
        # 遍巡整個資料集
        for data in (dataloader):
            # 將所有 tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            
            
            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors1, segments_tensors1, masks_tensors1 = data[:3]
            outputs1 = model(input_ids=tokens_tensors1, 
                            token_type_ids=segments_tensors1, 
                            attention_mask=masks_tensors1)
            
            
            Softmax = nn.Softmax(dim = 1)

            logits1= outputs1
            logits1 = Softmax(outputs1)
            _, pred1 = torch.max(logits1.data, 1)

            # 用來計算訓練集的分類準確率
            label1s = data[3]
            label2s = data[4]

            for i in range(pred1.size()[0]):
                y_true.append(label1s[i].item())
                y_pred.append(logits1[i][1].item())
                score_board.iloc[score_board_index, 10] = logits1[i][1].item()
                pred_label.append(pred1[i].item())
                score_board.iloc[score_board_index, 3] = logits1[i][1].item() >0.5
                    
                res2_score = logits1[i][1].item() / logits1[i][0].item()
                res2_score = max(res2_score, 1e-7)
                res2_score = min(res2_score, 1-1e-7)
                score_board.iloc[score_board_index, 5] = (math.log(res2_score) + shift_num)
                    
                if score_board.iloc[score_board_index, 2] == 1:
                    if score_board.iloc[score_board_index, 3] == 1: 
                        score_board.iloc[score_board_index, 6] = 1
                    else:
                        score_board.iloc[score_board_index, 6] = -1
                else:
                    score_board.iloc[score_board_index, 6] = 0
                   
                score_board_index += 1

#        
               
    d = {'y_true': y_true, 'pred_label': pred_label, 'y_pred': y_pred}
    df = pd.DataFrame(data=d)
    
    #calculate threshold
    val_times = 100
    acc = 0
    mean_thres = 0
    val_df = score_board[(score_board.y_true != 'other')].copy()
    for i in range(len(val_df)):
        if val_df.iloc[i, 8] == 'hyper':
            val_df.iloc[i, 8] = 1
        else:
            val_df.iloc[i, 8] = 0
    
    
    
    val_df.sort_values(by = 'task2_prob', ascending=False, inplace=True)
    fpr, tpr, threshold = metrics.roc_curve(val_df["y_true"].tolist(), val_df["task2_prob"].to_list())
    roc_auc = metrics.auc(fpr, tpr)

    
    correct = (df['y_true'] == df['pred_label']).sum()
    total = len(df)
    
    
    f1 = metrics.f1_score(df["y_true"], df["pred_label"])
        
    print('task2 ROC AUC:', roc_auc)
    return predictions, acc, roc_auc, correct, total, f1


print('BIBLESS', '-'*minus_num)
path = '../../data/evaluate/lexical_entailment/task2/BLESS/ABIBLESS.txt'
testset1 = HeirachicalDataset(tokenizer=tokenizer)
testset2 = DirectionDataset(tokenizer=tokenizer)

testloader1 = DataLoader(testset1, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch, shuffle = False)

testloader2 = DataLoader(testset2, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch, shuffle = False)



model_path = args.model1_path
state_dict = torch.load(model_path)
model_fine = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS, state_dict=state_dict)
model_fine.eval()
model_fine = model_fine.to(device)
_, acc, roc_auc, correct, total, f1 = get_predictions1(model_fine, testloader1)
del model_fine


model_path = args.model2_path
    
state_dict = torch.load(model_path)
model_fine = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS, state_dict=state_dict)
model_fine.eval()
model_fine = model_fine.to(device)
_, acc, roc_auc, correct, total, f1 = get_predictions2(model_fine, testloader2)
del model_fine


temp_other = score_board[(score_board.task1_res == 0) & (score_board.y_true == 'other')]
temp_hyper = score_board[(score_board.task1_res == 1) & (score_board.task2_res == 1) & (score_board.y_true == 'hyper')]
temp_rhyper = score_board[(score_board.task1_res == 1) & (score_board.task2_res == 0) & (score_board.y_true == 'rhyper')]
print("Accuracy:", (len(temp_other)+len(temp_hyper)+len(temp_rhyper)) / len(score_board))
print('')

for i in range(len(score_board)):
    if score_board.iloc[i, 8] == 'hyper':
        score_board.iloc[i, 8] = 1
    elif score_board.iloc[i, 8] == 'rhyper':
        score_board.iloc[i, 8] = -1
    else:
        score_board.iloc[i, 8] = 0



print('precision:')
print('micro:', metrics.precision_score(score_board['y_true'].tolist(), score_board['y_pred'].tolist(),  average='micro'))
print('macro:', metrics.precision_score(score_board['y_true'].tolist(), score_board['y_pred'].tolist(),  average='macro'))
print('weighted:', metrics.precision_score(score_board['y_true'].tolist(), score_board['y_pred'].tolist(),  average='weighted'))
print('')

print('recall:')
print('micro:', metrics.recall_score(score_board['y_true'].tolist(), score_board['y_pred'].tolist(),  average='micro'))
print('macro:', metrics.recall_score(score_board['y_true'].tolist(), score_board['y_pred'].tolist(),  average='macro'))
print('weighted:', metrics.recall_score(score_board['y_true'].tolist(), score_board['y_pred'].tolist(),  average='weighted'))
print('')

print('f1:')
print('micro:', metrics.f1_score(score_board['y_true'].tolist(), score_board['y_pred'].tolist(),  average='micro'))
print('macro:', metrics.f1_score(score_board['y_true'].tolist(), score_board['y_pred'].tolist(),  average='macro'))
print('weighted:', metrics.f1_score(score_board['y_true'].tolist(), score_board['y_pred'].tolist(),  average='weighted'))
print('')


print('HyperLex', '-'*20)
class hyperlexDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, tokenizer, task = 1):
        self.df  = pd.read_csv('../../data/evaluate/lexical_entailment/task2/hyperlex/hyperlex_formatted.txt', sep="\t", header=None)
        self.len = len(self.df)
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer
        self.task = task
    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        _, text_a, text_b , score6, score10= self.df.iloc[idx, :].values
        # 將 label 文字也轉換成索引方便轉換成 tensor

        # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
        word_pieces = ["[CLS]"]


        fake_a = ""
        tokens_a = self.tokenizer.tokenize(fake_a)
        fake_b = ""
        tokens_b = self.tokenizer.tokenize(fake_b)
        try:
            tokens_a = self.tokenizer.tokenize(text_a)
        except:
            print(idx, text_a, text_b)
            exit()
        
        try:
            tokens_b = self.tokenizer.tokenize(text_b)
        except:
            print(idx, text_a, text_b)
            exit()
    
        middle = "and"
        token_middle = self.tokenizer.tokenize(middle)
        
        tail = "are hierarchically related"
        token_tail = self.tokenizer.tokenize(tail)
        
        tail2 = "is a type of "
        token_tail2 = self.tokenizer.tokenize(tail2)


        if self.task == 1:
            word_pieces += (tokens_a + token_middle + tokens_b + token_tail + ["[SEP]"])
        elif self.task == 2:
            word_pieces += (tokens_a + token_tail2 + tokens_b + ["[SEP]"])
        len_a = len(word_pieces)

        tensor_score6 = torch.tensor(float(score6))
        tensor_score10 = torch.tensor(score10)
            
        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a, 
                                        dtype=torch.long)

        return (tokens_tensor, segments_tensor, tensor_score6, tensor_score10)
    
    def __len__(self):
        return self.len



def create_mini_batch_hyper(samples):
    tokens_tensors1 = [s[0] for s in samples]
    segments_tensors1 = [s[1] for s in samples]

    
    # 測試集有 labels
    if samples[0][2] is not None:
        label1_ids = torch.stack([s[2] for s in samples])
    else:
        label1_ids = None
    
    if samples[0][3] is not None:
        label2_ids = torch.stack([s[3] for s in samples])

    # zero pad 到同一序列長度
    tokens_tensors1 = pad_sequence(tokens_tensors1, 
                                  batch_first=True)
    segments_tensors1 = pad_sequence(segments_tensors1, 
                                    batch_first=True)
    

    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors1 = torch.zeros(tokens_tensors1.shape, 
                                dtype=torch.long)
    masks_tensors1 = masks_tensors1.masked_fill(
        tokens_tensors1 != 0, 1)

    return tokens_tensors1, segments_tensors1, masks_tensors1, label1_ids, label2_ids


testset_Hyper1 = hyperlexDataset(tokenizer=tokenizer, task = 1)
testset_Hyper2 = hyperlexDataset(tokenizer=tokenizer, task = 2)


testloader_Hyper1 = DataLoader(testset_Hyper1, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch_hyper, shuffle = False)
testloader_Hyper2 = DataLoader(testset_Hyper2, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch_hyper, shuffle = False)

hyper  = pd.read_csv('../../data/evaluate/lexical_entailment/task2/hyperlex/hyperlex_formatted.txt', sep="\t", header=None)

hyperboard_Col = ['task1_score', 'task2_score', 'rank_score', 'task2_prob']
hyperboard = pd.DataFrame(columns = hyperboard_Col)

hyperboard.insert(0, 'word1', hyper[1])
hyperboard.insert(1, 'word2', hyper[2])

def hyper_predictions(model1, model2, dataloader1, dataloader2):
    
    predictions = []
    single_guess = 1
    correct = 0
    fun_correct = 0
    total = 1
    y_pred = []
    y_true = []
    pred_label = []
    
    hyper_board_Col = ['task1_score', 'task2_score', 'rank_score']
    hyper_board = pd.DataFrame(columns = hyper_board_Col)
    hyperboard_index = 0
    shift_num = abs(math.log(1e-7, 10))

    with torch.no_grad():
        # 遍巡整個資料集
        for data in (dataloader1):
            # 將所有 tensors 移到 GPU 上
            data = [t.to("cuda:0") for t in data if t is not None]
            
            
            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors1, segments_tensors1, masks_tensors1 = data[:3]
            outputs1 = model1(input_ids=tokens_tensors1, 
                            token_type_ids=segments_tensors1, 
                            attention_mask=masks_tensors1)
            
            
            Softmax = nn.Softmax(dim = 1)

            logits1= outputs1
            logits1 = Softmax(outputs1)
            _, pred1 = torch.max(logits1.data, 1)

            for i in range(pred1.size()[0]):
                y_pred.append(logits1[i][1].item())
                pred_label.append(pred1[i].item())
                    
                hyperboard.iloc[hyperboard_index, 2] = math.log((logits1[i][1].item() / logits1[i][0].item()), 10)
                    
                    
                hyperboard_index += 1
    
    
        hyperboard_index = 0
        for data in (dataloader2):
            # 將所有 tensors 移到 GPU 上
            data = [t.to("cuda:0") for t in data if t is not None]
            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors1, segments_tensors1, masks_tensors1 = data[:3]
            #model2
            outputs2 = model2(input_ids=tokens_tensors1, 
                token_type_ids=segments_tensors1, 
                attention_mask=masks_tensors1)
            
            
            Softmax = nn.Softmax(dim = 1)

            logits2= outputs2
            logits2 = Softmax(outputs2)
            _, pred2 = torch.max(logits2.data, 1)

            for i in range(pred2.size()[0]):
                res2_score = logits2[i][1].item() / logits2[i][0].item()
                res2_score = max(res2_score, 1e-7)
                res2_score = min(res2_score, 1-1e-7)
                hyperboard.iloc[hyperboard_index, 3] = (math.log(res2_score, 10) + shift_num)
                hyperboard_index += 1

            # 用來計算訓練集的分類準確率
            label1s = data[3].cpu()
            label2s = data[4].cpu()
#                 total += labels.size(0)

    hyperboard['rank_score'] = hyperboard['task1_score'] + hyperboard['task2_score']

    acc = correct / total
    return predictions, total
    

model1_path = args.model1_path
model2_path = args.model2_path

state_dict = torch.load(model1_path)
model1_fine = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS, state_dict=state_dict)
model1_fine.eval()
model1_fine = model1_fine.to(device)

state_dict = torch.load(model2_path)
model2_fine = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS, state_dict=state_dict)
model2_fine.eval()
model2_fine = model2_fine.to(device)

_, acc = hyper_predictions(model1_fine, model2_fine, testloader_Hyper1, testloader_Hyper2)
del model1_fine
del model2_fine
torch.cuda.empty_cache()


df_hyper  = pd.read_csv('../../data/evaluate/lexical_entailment/task2/hyperlex/hyperlex_formatted.txt', sep="\t", header=None)

    
correlation, p = spearmanr(df_hyper[3].to_numpy(), hyperboard['task1_score'].to_numpy())
print('task1', 'correlation:', correlation, 'p:', p)
    
correlation, p = spearmanr(df_hyper[4].to_numpy(), hyperboard['task2_score'].to_numpy())
print('task2', 'correlation:', correlation, 'p:', p)

correlation, p = spearmanr(df_hyper[4].to_numpy(), hyperboard['rank_score'].to_numpy())
print('task1+2', 'correlation:', correlation, 'p:', p)
print('')


print('link reconstruction', '-'*minus_num)
class wordnetDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, tokenizer, task = 1):
        self.df  = pd.read_csv('../../data/evaluate/tree_testset.txt', sep="\t")
        self.len = len(self.df)
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer
        self.task = task
    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        text_a, text_b , weight = self.df.iloc[idx, :].values
        # 將 label 文字也轉換成索引方便轉換成 tensor

        # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
        word_pieces = ["[CLS]"]


        fake_a = ""
        tokens_a = self.tokenizer.tokenize(fake_a)
        fake_b = ""
        tokens_b = self.tokenizer.tokenize(fake_b)
        try:
            tokens_a = self.tokenizer.tokenize(text_a)
        except:
            print(idx, text_a, text_b)
            exit()
        
        try:
            tokens_b = self.tokenizer.tokenize(text_b)
        except:
            print(idx, text_a, text_b)
            exit()
    
        middle = "and"
        token_middle = self.tokenizer.tokenize(middle)
        
        tail = "are hierarchically related"
        token_tail = self.tokenizer.tokenize(tail)
        
        tail2 = "is a type of "
        token_tail2 = self.tokenizer.tokenize(tail2)


        if self.task == 1:
            word_pieces += (tokens_a + token_middle + tokens_b + token_tail + ["[SEP]"])
        elif self.task == 2:
            word_pieces += (tokens_a + token_tail2 + tokens_b + ["[SEP]"])
        len_a = len(word_pieces)

        if weight == 1:
            weight_tensor = torch.tensor(1)
        else:
            weight_tensor = torch.tensor(0)
        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a, 
                                        dtype=torch.long)


  

        return (tokens_tensor, segments_tensor, weight_tensor)
    
    def __len__(self):
        return self.len


def create_mini_batch_hyper(samples):
    tokens_tensors1 = [s[0] for s in samples]
    segments_tensors1 = [s[1] for s in samples]

    
    # 測試集有 labels
    if samples[0][2] is not None:
        label1_ids = torch.stack([s[2] for s in samples])
    else:
        label1_ids = None

    # zero pad 到同一序列長度
    tokens_tensors1 = pad_sequence(tokens_tensors1, 
                                  batch_first=True)
    segments_tensors1 = pad_sequence(segments_tensors1, 
                                    batch_first=True)
    

    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors1 = torch.zeros(tokens_tensors1.shape, 
                                dtype=torch.long)
    masks_tensors1 = masks_tensors1.masked_fill(
        tokens_tensors1 != 0, 1)

    return tokens_tensors1, segments_tensors1, masks_tensors1, label1_ids



testset_Wordnet1 = wordnetDataset(tokenizer=tokenizer, task = 1)
testset_Wordnet2 = wordnetDataset(tokenizer=tokenizer, task = 2)

testloader_Wordnet1 = DataLoader(testset_Wordnet1, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch_hyper, shuffle = False)
testloader_Wordnet2 = DataLoader(testset_Wordnet2, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch_hyper, shuffle = False)


wordnet_toxanomy  = pd.read_csv('../../data/evaluate/tree_testset.txt', sep="\t")
wordnet_toxanomy.insert(3, 'task1_res', None)
wordnet_toxanomy.insert(4, 'task2_res', None)
wordnet_toxanomy.insert(5, 'final_res', None)



def wordnet_predictions(model1, model2, dataloader1, dataloader2, wordnet_toxanomy = None):
    predictions = []
    single_guess = 1
    correct = 0
    fun_correct = 0
    total = 0
    y_pred = []
    y_true = []
    pred_label = []
    wordnet_toxanomy_index = 0

    with torch.no_grad():
        # 遍巡整個資料集
        for data in (dataloader1):
            # 將所有 tensors 移到 GPU 上
            data = [t.to("cuda:0") for t in data if t is not None]
            
            
            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors1, segments_tensors1, masks_tensors1 = data[:3]
            outputs1 = model1(input_ids=tokens_tensors1, 
                            token_type_ids=segments_tensors1, 
                            attention_mask=masks_tensors1)
            
            
            Softmax = nn.Softmax(dim = 1)

            logits1= outputs1
            logits1 = Softmax(outputs1)
            _, pred1 = torch.max(logits1.data, 1)
            for i in range(pred1.size()[0]):
                wordnet_toxanomy.iloc[wordnet_toxanomy_index, 3] = pred1[i].item()
                wordnet_toxanomy_index += 1
    
    
        wordnet_toxanomy_index = 0
        for data in (dataloader2):
            # 將所有 tensors 移到 GPU 上
            data = [t.to("cuda:0") for t in data if t is not None]
            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors1, segments_tensors1, masks_tensors1 = data[:3]
            #model2
            outputs2 = model2(input_ids=tokens_tensors1, 
                token_type_ids=segments_tensors1, 
                attention_mask=masks_tensors1)
            
            
            Softmax = nn.Softmax(dim = 1)

            logits2= outputs2
            logits2 = Softmax(outputs2)
            _, pred2 = torch.max(logits2.data, 1)

            for i in range(pred2.size()[0]):
                total+=1
                wordnet_toxanomy.iloc[wordnet_toxanomy_index, 4] = pred2[i].item()
                if wordnet_toxanomy.iloc[wordnet_toxanomy_index, 3] and wordnet_toxanomy.iloc[wordnet_toxanomy_index, 4]:
                    wordnet_toxanomy.iloc[wordnet_toxanomy_index, 5] = 1
                else:
                    wordnet_toxanomy.iloc[wordnet_toxanomy_index, 5] = 0
                
                if wordnet_toxanomy.iloc[wordnet_toxanomy_index, 5] == wordnet_toxanomy.iloc[wordnet_toxanomy_index, 2]:
                    correct+=1
                wordnet_toxanomy_index += 1
            # 用來計算訓練集的分類準確率
            label1s = data[3].cpu()


    acc = correct / total
    return predictions, acc
    


model1_path = args.model1_path
model2_path = args.model2_path

state_dict = torch.load(model1_path)
model1_fine = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS, state_dict=state_dict)
model1_fine.eval()
model1_fine = model1_fine.to(device)

state_dict = torch.load(model2_path)
model2_fine = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS, state_dict=state_dict)
model2_fine.eval()
model2_fine = model2_fine.to(device)


_, acc = wordnet_predictions(model1_fine, model2_fine, testloader_Wordnet1, testloader_Wordnet2, 
                             wordnet_toxanomy = wordnet_toxanomy)
del model1_fine
del model2_fine
torch.cuda.empty_cache()

print('Accuracy:', acc)

#draw tree
import pandas
import networkx as nx
from pyvis.network import Network

G = nx.Graph()
for i in range(len(wordnet_toxanomy)):
    G.add_node(wordnet_toxanomy['id1'][i])
    G.add_node(wordnet_toxanomy['id2'][i])
    if wordnet_toxanomy['final_res'][i] == 1:
        G.add_edge(wordnet_toxanomy['id1'][i], wordnet_toxanomy['id2'][i])


net = Network('1000px', '1000px', notebook = True)
net.from_nx(G)
net.show('../../output/wordnet_reconstruction_Q.html')
print('')