#!/usr/bin/env python
# coding: utf-8
from argparse import ArgumentParser

import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from sklearn import metrics
import math

from scipy.stats import spearmanr
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertForSequenceClassification
torch.no_grad()
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


parser = ArgumentParser()
parser.add_argument("-_model1_path", help="path to model", dest="model1_path", default = '../../output/task1_posneg_18.pth')
parser.add_argument("-_model2_path", help="path to model", dest="model2_path", default = '../../output/task2_posneg_6.pth')
args = parser.parse_args()


PRETRAINED_MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
NUM_LABELS = 2
minus_num = 20
BATCH_SIZE = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)




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

        middle_words = "is a type of"
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
        
        tokens_middle = self.tokenizer.tokenize(middle_words)

        try:
            tokens_b = self.tokenizer.tokenize(text_b)
        except:
            print(idx, text_a, text_b)
            exit()
        
        if(label1 == True):
            label1_tensor = torch.tensor(1)
        else:
            label1_tensor = torch.tensor(0)
        
        word_pieces += (tokens_a + tokens_middle + tokens_b + ["[SEP]"])
        len_a = len(word_pieces)

        y_pred1 = "positive"
        tokens_y_pred1 = self.tokenizer.tokenize(y_pred1)
        
        word_pieces1 = (word_pieces + tokens_y_pred1 + ["[SEP]"])
        len_b1 = len(word_pieces1) - len_a
 
        # 將整個 token 序列轉換成索引序列
        ids1 = self.tokenizer.convert_tokens_to_ids(word_pieces1)
        tokens_tensor1 = torch.tensor(ids1)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor1 = torch.tensor([0] * len_a + [1] * len_b1, 
                                        dtype=torch.long)


        y_pred2 = "negative"
        tokens_y_pred2 = self.tokenizer.tokenize(y_pred2)
        
        word_pieces2 = (word_pieces + tokens_y_pred2 + ["[SEP]"])
        len_b2 = len(word_pieces2) - len_a
  
        # 將整個 token 序列轉換成索引序列
        ids2 = self.tokenizer.convert_tokens_to_ids(word_pieces2)
        tokens_tensor2 = torch.tensor(ids2)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor2 = torch.tensor([0] * len_a + [1] * len_b2, 
                                        dtype=torch.long)

        
        
        # 將 label 文字也轉換成索引方便轉換成 tensor

        return (tokens_tensor1, segments_tensor1, tokens_tensor2, segments_tensor2, label1_tensor)
    
    def __len__(self):
        return self.len




path = "../../data/evaluate/lexical_entailment/task1/bless2011/data.tsv"
testset2 = BlessDataset(tokenizer=tokenizer)

def create_mini_batch_bless(samples):
    tokens_tensors1 = [s[0] for s in samples]
    segments_tensors1 = [s[1] for s in samples]

    tokens_tensors2 = [s[2] for s in samples]
    segments_tensors2 = [s[3] for s in samples]
    
    # 測試集有 labels
    if samples[0][4] is not None:
        label_ids = torch.stack([s[4] for s in samples])
    else:
        label_ids = None
    
    # zero pad 到同一序列長度
    tokens_tensors1 = pad_sequence(tokens_tensors1, 
                                  batch_first=True)
    segments_tensors1 = pad_sequence(segments_tensors1, 
                                    batch_first=True)
    
    tokens_tensors2 = pad_sequence(tokens_tensors2, 
                                  batch_first=True)
    segments_tensors2 = pad_sequence(segments_tensors2, 
                                    batch_first=True)

    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors1 = torch.zeros(tokens_tensors1.shape, 
                                dtype=torch.long)
    masks_tensors1 = masks_tensors1.masked_fill(
        tokens_tensors1 != 0, 1)
    
    
    masks_tensors2 = torch.zeros(tokens_tensors2.shape, 
                                dtype=torch.long)
    masks_tensors2 = masks_tensors2.masked_fill(
        tokens_tensors2 != 0, 1)

    
    return tokens_tensors1, segments_tensors1, masks_tensors1, tokens_tensors2, segments_tensors2, masks_tensors2, label_ids


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
            
            
            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors1, segments_tensors1, masks_tensors1, tokens_tensors2, segments_tensors2, masks_tensors2 = data[:6]
            
            outputs1 = model(input_ids=tokens_tensors1, 
                            token_type_ids=segments_tensors1, 
                            attention_mask=masks_tensors1)
            outputs2 = model(input_ids=tokens_tensors2, 
                            token_type_ids=segments_tensors2, 
                            attention_mask=masks_tensors2)

            
            Softmax = nn.Softmax(dim = 1)

            logits1 = outputs1
            logits1 = Softmax(outputs1)
            
            logits2 = outputs2
            logits2 = Softmax(outputs2)

            _, pred1 = torch.max(logits1.data, 1)

            label1s = data[3]

            for i in range(pred1.size()[0]):
                if logits1[i][1] > logits2[i][1]:
                    score_board.iloc[score_board_index, 3] = 1
                    score_board.iloc[score_board_index, 10] = logits1[i][1].item()
                    if score_board.iloc[score_board_index, 8]:
                        correct+=1
                else:
                    score_board.iloc[score_board_index, 3] = 0
                    score_board.iloc[score_board_index, 10] = (1 - logits2[i][1].item())
                score_board_index+=1
    val_df = score_board[(score_board.y_true == True)].copy()

    total = len(val_df)
    acc = correct / total
    print("Accuracy:" , acc)
    
    return acc
    


model_path = args.model2_path
    
state_dict = torch.load(model_path)
model_fine = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS, state_dict=state_dict)
model_fine.eval()
model_fine = model_fine.to(device)
acc = get_prediction_bless(model_fine, testloader2)
del model_fine
torch.cuda.empty_cache()
print('')


print('BIBLESS', '-'*minus_num)
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
        middle_words = "and"
        tail_words = "are hierarchically related"

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
        
        tokens_middle = self.tokenizer.tokenize(middle_words)
        tokens_tail = self.tokenizer.tokenize(tail_words)

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

        
        word_pieces += (tokens_a + tokens_middle + tokens_b + tokens_tail + ["[SEP]"])
        len_a = len(word_pieces)

        y_pred1 = "positive"
        tokens_y_pred1 = self.tokenizer.tokenize(y_pred1)
        
        word_pieces1 = (word_pieces + tokens_y_pred1 + ["[SEP]"])
        len_b1 = len(word_pieces1) - len_a
 
        # 將整個 token 序列轉換成索引序列
        ids1 = self.tokenizer.convert_tokens_to_ids(word_pieces1)
        tokens_tensor1 = torch.tensor(ids1)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor1 = torch.tensor([0] * len_a + [1] * len_b1, 
                                        dtype=torch.long)


        y_pred2 = "negative"
        tokens_y_pred2 = self.tokenizer.tokenize(y_pred2)
        
        word_pieces2 = (word_pieces + tokens_y_pred2 + ["[SEP]"])
        len_b2 = len(word_pieces2) - len_a
  
        # 將整個 token 序列轉換成索引序列
        ids2 = self.tokenizer.convert_tokens_to_ids(word_pieces2)
        tokens_tensor2 = torch.tensor(ids2)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor2 = torch.tensor([0] * len_a + [1] * len_b2, 
                                        dtype=torch.long)

        
        
        # 將 label 文字也轉換成索引方便轉換成 tensor

        return (tokens_tensor1, segments_tensor1, tokens_tensor2, segments_tensor2, label1_tensor, label2_tensor)
    
    def __len__(self):
        return self.len
    
    


class DirectionDataset(Dataset):
     # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, tokenizer):
#         assert mode in ["train", "test"]  # 一般訓練你會需要 dev set
#         self.mode = mode
        # 大數據你會需要用 iterator=True
        self.df  = pd.read_csv(path, sep="\t", header=None)
        self.len = len(self.df)
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer

    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        text_a, text_b , label1, label2= self.df.iloc[idx, :].values
        # 將 label 文字也轉換成索引方便轉換成 tensor
        middle_words = "is a type of"

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
        
        tokens_middle = self.tokenizer.tokenize(middle_words)

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

        
        word_pieces += (tokens_a + tokens_middle + tokens_b + ["[SEP]"])
        len_a = len(word_pieces)

        y_pred1 = "positive"
        tokens_y_pred1 = self.tokenizer.tokenize(y_pred1)
        
        word_pieces1 = (word_pieces + tokens_y_pred1 + ["[SEP]"])
        len_b1 = len(word_pieces1) - len_a
 
        # 將整個 token 序列轉換成索引序列
        ids1 = self.tokenizer.convert_tokens_to_ids(word_pieces1)
        tokens_tensor1 = torch.tensor(ids1)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor1 = torch.tensor([0] * len_a + [1] * len_b1, 
                                        dtype=torch.long)


        y_pred2 = "negative"
        tokens_y_pred2 = self.tokenizer.tokenize(y_pred2)
        
        word_pieces2 = (word_pieces + tokens_y_pred2 + ["[SEP]"])
        len_b2 = len(word_pieces2) - len_a
  
        # 將整個 token 序列轉換成索引序列
        ids2 = self.tokenizer.convert_tokens_to_ids(word_pieces2)
        tokens_tensor2 = torch.tensor(ids2)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor2 = torch.tensor([0] * len_a + [1] * len_b2, 
                                        dtype=torch.long)
        
        # 將 label 文字也轉換成索引方便轉換成 tensor
        return (tokens_tensor1, segments_tensor1, tokens_tensor2, segments_tensor2, label1_tensor, label2_tensor)
    
    def __len__(self):
        return self.len
    
    
    

path = "../../data/evaluate/lexical_entailment/task2/BLESS/ABIBLESS.txt"

testset1 = HeirachicalDataset(tokenizer=tokenizer)
testset2 = DirectionDataset(tokenizer=tokenizer)



def create_mini_batch(samples):
    tokens_tensors1 = [s[0] for s in samples]
    segments_tensors1 = [s[1] for s in samples]

    tokens_tensors2 = [s[2] for s in samples]
    segments_tensors2 = [s[3] for s in samples]
    
    # 測試集有 labels
    if samples[0][4] is not None:
        label_ids = torch.stack([s[4] for s in samples])
    else:
        label_ids = None
    
    # zero pad 到同一序列長度
    tokens_tensors1 = pad_sequence(tokens_tensors1, 
                                  batch_first=True)
    segments_tensors1 = pad_sequence(segments_tensors1, 
                                    batch_first=True)
    
    tokens_tensors2 = pad_sequence(tokens_tensors2, 
                                  batch_first=True)
    segments_tensors2 = pad_sequence(segments_tensors2, 
                                    batch_first=True)

    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors1 = torch.zeros(tokens_tensors1.shape, 
                                dtype=torch.long)
    masks_tensors1 = masks_tensors1.masked_fill(
        tokens_tensors1 != 0, 1)
    
    
    masks_tensors2 = torch.zeros(tokens_tensors2.shape, 
                                dtype=torch.long)
    masks_tensors2 = masks_tensors2.masked_fill(
        tokens_tensors2 != 0, 1)

    
    return tokens_tensors1, segments_tensors1, masks_tensors1, tokens_tensors2, segments_tensors2, masks_tensors2, label_ids


testloader1 = DataLoader(testset1, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch, shuffle = False)

testloader2 = DataLoader(testset2, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch, shuffle = False)


bibless  = pd.read_csv(path, sep="\t", header=None)

score_board_Col = ['task1_res', 'task2_res', 'task1_score', 'task2_score', 'y_pred', 'rank_score','task1_prob', 'task2_prob']
score_board = pd.DataFrame(columns = score_board_Col)

score_board.insert(0, 'word1', bibless[0])
score_board.insert(1, 'word2', bibless[1])
score_board.insert(8, 'y_true', bibless[3])


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
            tokens_tensors1, segments_tensors1, masks_tensors1, tokens_tensors2, segments_tensors2, masks_tensors2 = data[:6]
            
            outputs1 = model(input_ids=tokens_tensors1, 
                            token_type_ids=segments_tensors1, 
                            attention_mask=masks_tensors1)
            outputs2 = model(input_ids=tokens_tensors2, 
                            token_type_ids=segments_tensors2, 
                            attention_mask=masks_tensors2)

            
            Softmax = nn.Softmax(dim = 1)

            logits1 = outputs1
            logits1 = Softmax(outputs1)
            
            logits2 = outputs2
            logits2 = Softmax(outputs2)

            _, pred1 = torch.max(logits1.data, 1)


            # 用來計算訓練集的分類準確率
            labels = data[6]

            for i in range(pred1.size()[0]):
                y_true.append(labels[i].item())
                    
                if logits1[i][1] > logits2[i][1]:
                        
                    y_pred.append(logits1[i][1].item())
                    score_board.iloc[score_board_index, 9] = logits1[i][1].item()
                    pred_label.append(1)
                        
                else:
                    y_pred.append( (1 - logits2[i][1].item()) )
                    score_board.iloc[score_board_index, 9] = (1 - logits2[i][1].item())
                    pred_label.append(0)
                        
                score_board.iloc[score_board_index, 2] = pred1[i].item()
                score_board.iloc[score_board_index, 4] = math.log(logits1[i][1].item() / logits2[i][1].item())
                score_board_index += 1


                    
               
    d = {'y_true': y_true, 'pred_label': pred_label, 'y_pred': y_pred}
    df = pd.DataFrame(data=d)
    df.sort_values(by = 'y_pred', ascending=False, inplace=True)

    
    correct = (df['y_true'] == df['pred_label']).sum()
    total = len(df)
    
    
    f1 = metrics.f1_score(df["y_true"], df["pred_label"])
    
    fpr, tpr, threshold = metrics.roc_curve(df["y_true"], df["y_pred"])
    roc_auc = metrics.auc(fpr, tpr)
    acc = 0
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
    shift_num = abs(math.log(1e-7, 10))

    
    with torch.no_grad():
        # 遍巡整個資料集
        for data in (dataloader):
            # 將所有 tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            
            
            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors1, segments_tensors1, masks_tensors1, tokens_tensors2, segments_tensors2, masks_tensors2 = data[:6]
            
            outputs1 = model(input_ids=tokens_tensors1, 
                            token_type_ids=segments_tensors1, 
                            attention_mask=masks_tensors1)
            outputs2 = model(input_ids=tokens_tensors2, 
                            token_type_ids=segments_tensors2, 
                            attention_mask=masks_tensors2)

            
            Softmax = nn.Softmax(dim = 1)

            logits1 = outputs1
            logits1 = Softmax(outputs1)
            
            logits2 = outputs2
            logits2 = Softmax(outputs2)

            _, pred1 = torch.max(logits1.data, 1)


            labels = data[6]

            for i in range(pred1.size()[0]):
                y_true.append(labels[i].item())
                    
                if logits1[i][1] > logits2[i][1]:
                    y_pred.append(logits1[i][1].item())
                    pred_label.append(1)
                    score_board.iloc[score_board_index, 3] = 1
                    score_board.iloc[score_board_index, 10] = logits1[i][1].item()
                        

                        
                else:
                    y_pred.append( (1 - logits2[i][1].item()) )
                    pred_label.append(0)
                    score_board.iloc[score_board_index, 3] = 0
                    score_board.iloc[score_board_index, 10] = (1 - logits2[i][1].item())

                    
                res2_score = logits1[i][1].item() / logits2[i][1].item()
                res2_score = max(res2_score, 1e-7)
                res2_score = min(res2_score, 1-1e-7)
                score_board.iloc[score_board_index, 5] = (math.log(res2_score, 10) + shift_num)
                    
                if score_board.iloc[score_board_index, 2] == 1:
                    if score_board.iloc[score_board_index, 3] == 1: 
                        score_board.iloc[score_board_index, 6] = 1
                    else:
                        score_board.iloc[score_board_index, 6] = -1
                else:
                    score_board.iloc[score_board_index, 6] = 0

                score_board_index += 1


                    
               
    d = {'y_true': y_true, 'pred_label': pred_label, 'y_pred': y_pred}
    df = pd.DataFrame(data=d)
    df.sort_values(by = 'y_pred', ascending=False, inplace=True)

    
    df['y_true']
    correct = (df['y_true'] == df['pred_label']).sum()
    total = len(df)
    
    val_df = score_board[(score_board.y_true != 'other')].copy()
    for i in range(len(val_df)):
        if val_df.iloc[i, 8] == 'hyper':
            val_df.iloc[i, 8] = 1
        else:
            val_df.iloc[i, 8] = 0
    
    
    
    val_df.sort_values(by = 'task2_prob', ascending=False, inplace=True)
    fpr, tpr, threshold = metrics.roc_curve(val_df["y_true"].tolist(), val_df["task2_prob"].to_list())
    roc_auc = metrics.auc(fpr, tpr)
    
    f1 = metrics.f1_score(df["y_true"], df["pred_label"])

    print('task2 ROC AUC:', roc_auc)
    return predictions, acc, roc_auc, correct, total, f1
    

path = "../../data/evaluate/lexical_entailment/task2/BLESS/ABIBLESS.txt"
testset = HeirachicalDataset(tokenizer=tokenizer)

model_path = args.model1_path

state_dict = torch.load(model_path)
model_fine = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS, state_dict=state_dict)
model_fine.eval()
model_fine = model_fine.to(device)
_, acc, roc_auc, correct, total, f1 = get_predictions1(model_fine, testloader1)
del model_fine
torch.cuda.empty_cache()

model_path = args.model2_path
    
state_dict = torch.load(model_path)
model_fine = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS, state_dict=state_dict)
model_fine.eval()
model_fine = model_fine.to(device)
_, acc, roc_auc, correct, total, f1 = get_predictions2(model_fine, testloader2)
del model_fine
torch.cuda.empty_cache()


temp_other = score_board[(score_board.task1_res == 0) & (score_board.y_true == 'other')]
temp_hyper = score_board[(score_board.task1_res == 1) & (score_board.task2_res == 1) & (score_board.y_true == 'hyper')]
temp_rhyper = score_board[(score_board.task1_res == 1) & (score_board.task2_res == 0) & (score_board.y_true == 'rhyper')]
print("accuracy: ", (len(temp_other)+len(temp_hyper)+len(temp_rhyper)) / len(score_board))
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



print('HyperLex', '-'*minus_num)
class hyperlexDataset1(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, tokenizer):
#         assert mode in ["train", "test"]  # 一般訓練你會需要 dev set
#         self.mode = mode
        # 大數據你會需要用 iterator=True
        self.df  = pd.read_csv('../../data/evaluate/lexical_entailment/task2/hyperlex/hyperlex_formatted.txt', sep="\t", header=None)
        self.len = len(self.df)
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer

    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        _, text_a, text_b , label1, label2= self.df.iloc[idx, :].values
        # 將 label 文字也轉換成索引方便轉換成 tensor
        middle_words = "and"
        tail_words = "are hierarchically related"

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
        
        tokens_middle = self.tokenizer.tokenize(middle_words)
        tokens_tail = self.tokenizer.tokenize(tail_words)

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

        
        word_pieces += (tokens_a + tokens_middle + tokens_b + tokens_tail + ["[SEP]"])
        len_a = len(word_pieces)

        y_pred1 = "positive"
        tokens_y_pred1 = self.tokenizer.tokenize(y_pred1)
        
        word_pieces1 = (word_pieces + tokens_y_pred1 + ["[SEP]"])
        len_b1 = len(word_pieces1) - len_a
 
        # 將整個 token 序列轉換成索引序列
        ids1 = self.tokenizer.convert_tokens_to_ids(word_pieces1)
        tokens_tensor1 = torch.tensor(ids1)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor1 = torch.tensor([0] * len_a + [1] * len_b1, 
                                        dtype=torch.long)


        y_pred2 = "negative"
        tokens_y_pred2 = self.tokenizer.tokenize(y_pred2)
        
        word_pieces2 = (word_pieces + tokens_y_pred2 + ["[SEP]"])
        len_b2 = len(word_pieces2) - len_a
  
        # 將整個 token 序列轉換成索引序列
        ids2 = self.tokenizer.convert_tokens_to_ids(word_pieces2)
        tokens_tensor2 = torch.tensor(ids2)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor2 = torch.tensor([0] * len_a + [1] * len_b2, 
                                        dtype=torch.long)

        
        
        # 將 label 文字也轉換成索引方便轉換成 tensor

        return (tokens_tensor1, segments_tensor1, tokens_tensor2, segments_tensor2)
    
    def __len__(self):
        return self.len
    
    


class hyperlexDataset2(Dataset):
     # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, tokenizer):
#         assert mode in ["train", "test"]  # 一般訓練你會需要 dev set
#         self.mode = mode
        # 大數據你會需要用 iterator=True
        self.df  = pd.read_csv('../../data/evaluate/lexical_entailment/task2/hyperlex/hyperlex_formatted.txt', sep="\t", header=None)
        self.len = len(self.df)
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer

    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        _, text_a, text_b , label1, label2 = self.df.iloc[idx, :].values
        # 將 label 文字也轉換成索引方便轉換成 tensor
        middle_words = "is a type of"

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
        
        tokens_middle = self.tokenizer.tokenize(middle_words)

        try:
            tokens_b = self.tokenizer.tokenize(text_b)
        except:
            print(idx, text_a, text_b)
            exit()
        
        word_pieces += (tokens_a + tokens_middle + tokens_b + ["[SEP]"])
        len_a = len(word_pieces)

        y_pred1 = "positive"
        tokens_y_pred1 = self.tokenizer.tokenize(y_pred1)
        
        word_pieces1 = (word_pieces + tokens_y_pred1 + ["[SEP]"])
        len_b1 = len(word_pieces1) - len_a
 
        # 將整個 token 序列轉換成索引序列
        ids1 = self.tokenizer.convert_tokens_to_ids(word_pieces1)
        tokens_tensor1 = torch.tensor(ids1)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor1 = torch.tensor([0] * len_a + [1] * len_b1, 
                                        dtype=torch.long)


        y_pred2 = "negative"
        tokens_y_pred2 = self.tokenizer.tokenize(y_pred2)
        
        word_pieces2 = (word_pieces + tokens_y_pred2 + ["[SEP]"])
        len_b2 = len(word_pieces2) - len_a
  
        # 將整個 token 序列轉換成索引序列
        ids2 = self.tokenizer.convert_tokens_to_ids(word_pieces2)
        tokens_tensor2 = torch.tensor(ids2)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor2 = torch.tensor([0] * len_a + [1] * len_b2, 
                                        dtype=torch.long)

        
        
        # 將 label 文字也轉換成索引方便轉換成 tensor
        return (tokens_tensor1, segments_tensor1, tokens_tensor2, segments_tensor2)    
    
    def __len__(self):
        return self.len


def create_mini_batch_hyper(samples):
    tokens_tensors1 = [s[0] for s in samples]
    segments_tensors1 = [s[1] for s in samples]

    tokens_tensors2 = [s[2] for s in samples]
    segments_tensors2 = [s[3] for s in samples]
        
    # zero pad 到同一序列長度
    tokens_tensors1 = pad_sequence(tokens_tensors1, 
                                  batch_first=True)
    segments_tensors1 = pad_sequence(segments_tensors1, 
                                    batch_first=True)
    
    tokens_tensors2 = pad_sequence(tokens_tensors2, 
                                  batch_first=True)
    segments_tensors2 = pad_sequence(segments_tensors2, 
                                    batch_first=True)

    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors1 = torch.zeros(tokens_tensors1.shape, 
                                dtype=torch.long)
    masks_tensors1 = masks_tensors1.masked_fill(
        tokens_tensors1 != 0, 1)
    
    
    masks_tensors2 = torch.zeros(tokens_tensors2.shape, 
                                dtype=torch.long)
    masks_tensors2 = masks_tensors2.masked_fill(
        tokens_tensors2 != 0, 1)

    
    return tokens_tensors1, segments_tensors1, masks_tensors1, tokens_tensors2, segments_tensors2, masks_tensors2


testset_Hyper1 = hyperlexDataset1(tokenizer=tokenizer)
testset_Hyper2 = hyperlexDataset2(tokenizer=tokenizer)

testloader_Hyper1 = DataLoader(testset_Hyper1, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch_hyper, shuffle = False)
testloader_Hyper2 = DataLoader(testset_Hyper2, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch_hyper, shuffle = False)





hyper  = pd.read_csv('../../data/evaluate/lexical_entailment/task2/hyperlex/hyperlex_formatted.txt', sep="\t", header=None)

hyperboard_Col = ['task1_score', 'task2_score', 'rank_score']
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

    return predictions, total
    return predictions
    
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
print('task1\t', 'correlation:', correlation, 'p:', p)
    
correlation, p = spearmanr(df_hyper[4].to_numpy(), hyperboard['task2_score'].to_numpy())
print('task2\t', 'correlation:', correlation, 'p:', p)

correlation, p = spearmanr(df_hyper[4].to_numpy(), hyperboard['rank_score'].to_numpy())
print('task1+2\t', 'correlation:', correlation, 'p:', p)
print('')

# # link reconstruction

# In[47]:

print('link reconstruction', '-'*minus_num)
class wordnetDataset1(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, tokenizer):
        self.df  = pd.read_csv('../../data/evaluate/tree_testset.txt', sep="\t")
        self.len = len(self.df)
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer

    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        text_a, text_b , label1= self.df.iloc[idx, :].values
        # 將 label 文字也轉換成索引方便轉換成 tensor
        middle_words = "and"
        tail_words = "are hierarchically related"

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
        
        tokens_middle = self.tokenizer.tokenize(middle_words)
        tokens_tail = self.tokenizer.tokenize(tail_words)

        try:
            tokens_b = self.tokenizer.tokenize(text_b)
        except:
            print(idx, text_a, text_b)
            exit()
        
        if(label1 == 1):
            label1_tensor = torch.tensor(1)
        else:
            label1_tensor = torch.tensor(0)
            

        
        word_pieces += (tokens_a + tokens_middle + tokens_b + tokens_tail + ["[SEP]"])
        len_a = len(word_pieces)

        y_pred1 = "positive"
        tokens_y_pred1 = self.tokenizer.tokenize(y_pred1)
        
        word_pieces1 = (word_pieces + tokens_y_pred1 + ["[SEP]"])
        len_b1 = len(word_pieces1) - len_a
 
        # 將整個 token 序列轉換成索引序列
        ids1 = self.tokenizer.convert_tokens_to_ids(word_pieces1)
        tokens_tensor1 = torch.tensor(ids1)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor1 = torch.tensor([0] * len_a + [1] * len_b1, 
                                        dtype=torch.long)


        y_pred2 = "negative"
        tokens_y_pred2 = self.tokenizer.tokenize(y_pred2)
        
        word_pieces2 = (word_pieces + tokens_y_pred2 + ["[SEP]"])
        len_b2 = len(word_pieces2) - len_a
  
        # 將整個 token 序列轉換成索引序列
        ids2 = self.tokenizer.convert_tokens_to_ids(word_pieces2)
        tokens_tensor2 = torch.tensor(ids2)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor2 = torch.tensor([0] * len_a + [1] * len_b2, 
                                        dtype=torch.long)

        
        
        # 將 label 文字也轉換成索引方便轉換成 tensor
        
        if weight ==1:
            label1_tensor = torch.tensor(1)
        else:
            label1_tensor = torch.tensor(0)
        return (tokens_tensor1, segments_tensor1, tokens_tensor2, segments_tensor2, label1_tensor)
    
    def __len__(self):
        return self.len
    

class wordnetDataset2(Dataset):
     # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, tokenizer):
        self.df  = pd.read_csv('../../data/evaluate/tree_testset.txt', sep="\t")
        self.len = len(self.df)
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer

    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        text_a, text_b , label1 = self.df.iloc[idx, :].values
        # 將 label 文字也轉換成索引方便轉換成 tensor
        middle_words = "is a type of"

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
        
        tokens_middle = self.tokenizer.tokenize(middle_words)

        try:
            tokens_b = self.tokenizer.tokenize(text_b)
        except:
            print(idx, text_a, text_b)
            exit()
        
        word_pieces += (tokens_a + tokens_middle + tokens_b + ["[SEP]"])
        len_a = len(word_pieces)

        y_pred1 = "positive"
        tokens_y_pred1 = self.tokenizer.tokenize(y_pred1)
        
        word_pieces1 = (word_pieces + tokens_y_pred1 + ["[SEP]"])
        len_b1 = len(word_pieces1) - len_a
 
        # 將整個 token 序列轉換成索引序列
        ids1 = self.tokenizer.convert_tokens_to_ids(word_pieces1)
        tokens_tensor1 = torch.tensor(ids1)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor1 = torch.tensor([0] * len_a + [1] * len_b1, 
                                        dtype=torch.long)


        y_pred2 = "negative"
        tokens_y_pred2 = self.tokenizer.tokenize(y_pred2)
        
        word_pieces2 = (word_pieces + tokens_y_pred2 + ["[SEP]"])
        len_b2 = len(word_pieces2) - len_a
  
        # 將整個 token 序列轉換成索引序列
        ids2 = self.tokenizer.convert_tokens_to_ids(word_pieces2)
        tokens_tensor2 = torch.tensor(ids2)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor2 = torch.tensor([0] * len_a + [1] * len_b2, 
                                        dtype=torch.long)

        label1_tensor = torch.tensor(1)
        
        # 將 label 文字也轉換成索引方便轉換成 tensor
        return (tokens_tensor1, segments_tensor1, tokens_tensor2, segments_tensor2, label1_tensor)    
    
    def __len__(self):
        return self.len


def create_mini_batch_wordnet(samples):
    tokens_tensors1 = [s[0] for s in samples]
    segments_tensors1 = [s[1] for s in samples]

    tokens_tensors2 = [s[2] for s in samples]
    segments_tensors2 = [s[3] for s in samples]
    if samples[0][4] is not None:
        label_ids = torch.stack([s[4] for s in samples])
    else:
        label_ids = None

    # zero pad 到同一序列長度
    tokens_tensors1 = pad_sequence(tokens_tensors1, 
                                  batch_first=True)
    segments_tensors1 = pad_sequence(segments_tensors1, 
                                    batch_first=True)
    
    tokens_tensors2 = pad_sequence(tokens_tensors2, 
                                  batch_first=True)
    segments_tensors2 = pad_sequence(segments_tensors2, 
                                    batch_first=True)

    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors1 = torch.zeros(tokens_tensors1.shape, 
                                dtype=torch.long)
    masks_tensors1 = masks_tensors1.masked_fill(
        tokens_tensors1 != 0, 1)
    
    
    masks_tensors2 = torch.zeros(tokens_tensors2.shape, 
                                dtype=torch.long)
    masks_tensors2 = masks_tensors2.masked_fill(
        tokens_tensors2 != 0, 1)

    
    return tokens_tensors1, segments_tensors1, masks_tensors1, tokens_tensors2, segments_tensors2, masks_tensors2, label_ids


testset_wordnet1 = wordnetDataset1(tokenizer=tokenizer)
testset_wordnet2 = wordnetDataset2(tokenizer=tokenizer)

testloader_Wordnet1 = DataLoader(testset_wordnet1, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch_wordnet, shuffle = False)
testloader_Wordnet2 = DataLoader(testset_wordnet2, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch_wordnet, shuffle = False)


wordnet_toxanomy  = pd.read_csv('../../data/evaluate/tree_testset.txt', sep="\t")
wordnet_toxanomy.insert(3, 'task1_res', None)
wordnet_toxanomy.insert(4, 'task2_res', None)
wordnet_toxanomy.insert(5, 'final_res', None)

def get_prediction_wordnet(model1, model2, dataloader1, dataloader2):
    correct = 0
    fun_correct = 0
    total = 0
    wordnet_toxanomy_index = 0
    
    with torch.no_grad():
        # 遍巡整個資料集
        for data in (dataloader1):
            # 將所有 tensors 移到 GPU 上
            if next(model1.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            
            
            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors1, segments_tensors1, masks_tensors1, tokens_tensors2, segments_tensors2, masks_tensors2 = data[:6]
            
            outputs1 = model1(input_ids=tokens_tensors1, 
                            token_type_ids=segments_tensors1, 
                            attention_mask=masks_tensors1)
            outputs2 = model1(input_ids=tokens_tensors2, 
                            token_type_ids=segments_tensors2, 
                            attention_mask=masks_tensors2)

            
            Softmax = nn.Softmax(dim = 1)

            logits1 = outputs1
            logits1 = Softmax(outputs1)
            
            logits2 = outputs2
            logits2 = Softmax(outputs2)

            _, pred1 = torch.max(logits1.data, 1)

            # 用來計算訓練集的分類準確率
            label1s = data[3]

            for i in range(pred1.size()[0]):
                if logits1[i][1] > logits2[i][1]:
                    wordnet_toxanomy.iloc[wordnet_toxanomy_index, 3] = 1
                else:
                    wordnet_toxanomy.iloc[wordnet_toxanomy_index, 3] = 0
                wordnet_toxanomy_index+=1

        wordnet_toxanomy_index = 0
        for data in (dataloader2):
            # 將所有 tensors 移到 GPU 上
            if next(model2.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            
            
            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors1, segments_tensors1, masks_tensors1, tokens_tensors2, segments_tensors2, masks_tensors2 = data[:6]
            
            outputs1 = model2(input_ids=tokens_tensors1, 
                            token_type_ids=segments_tensors1, 
                            attention_mask=masks_tensors1)
            outputs2 = model2(input_ids=tokens_tensors2, 
                            token_type_ids=segments_tensors2, 
                            attention_mask=masks_tensors2)

            
            Softmax = nn.Softmax(dim = 1)

            logits1 = outputs1
            logits1 = Softmax(outputs1)
            
            logits2 = outputs2
            logits2 = Softmax(outputs2)

            _, pred1 = torch.max(logits1.data, 1)

            # 用來計算訓練集的分類準確率
            label1s = data[3]

            for i in range(pred1.size()[0]):
                total+=1
                if logits1[i][1] > logits2[i][1]:
                    wordnet_toxanomy.iloc[wordnet_toxanomy_index, 4] = 1
                else:
                    wordnet_toxanomy.iloc[wordnet_toxanomy_index, 4] = 0
                    
                if wordnet_toxanomy.iloc[wordnet_toxanomy_index, 3] and wordnet_toxanomy.iloc[wordnet_toxanomy_index, 4]:
                    wordnet_toxanomy.iloc[wordnet_toxanomy_index, 5] = 1
                else:
                    wordnet_toxanomy.iloc[wordnet_toxanomy_index, 5] = 0
                if wordnet_toxanomy.iloc[wordnet_toxanomy_index, 5] == wordnet_toxanomy.iloc[wordnet_toxanomy_index, 2]:
                    correct+=1
                wordnet_toxanomy_index+=1

    acc = correct / total
    return acc

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


acc = get_prediction_wordnet(model1_fine, model2_fine, testloader_Wordnet2, testloader_Wordnet2)
del model1_fine
del model2_fine
torch.cuda.empty_cache()
print('Accuracy: ', acc)


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
net.show('../../output/wordnet_reconstruction_PosNeg.html')
print('')
