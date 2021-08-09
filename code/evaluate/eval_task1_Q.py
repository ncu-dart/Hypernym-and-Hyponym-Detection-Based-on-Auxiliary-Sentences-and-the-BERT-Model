#!/usr/bin/env python
# coding: utf-8
from argparse import ArgumentParser

import torch
import torch.nn as nn
import pandas as pd
from sklearn import metrics
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertForSequenceClassification
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


parser = ArgumentParser()
parser.add_argument("-model_path", help="path to model", dest="model_path", default = '../../output/task1_BERT_0.pth')
args = parser.parse_args()


PRETRAINED_MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
NUM_LABELS = 2


class HeirachicalDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, tokenizer):
        self.df  = pd.read_csv(path, sep="\t", header=None)
        self.len = len(self.df)
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer

    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        text_a, text_b , label= self.df.iloc[idx, :].values
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

        if(label == True):
            label_tensor = torch.tensor(1)
        else:
            label_tensor = torch.tensor(0)

        
        word_pieces += (tokens_a + tokens_middle + tokens_b + tokens_tail + ["[SEP]"])
        len_all = len(word_pieces)

        
        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0]* len_all, 
                                        dtype=torch.long)
        
        
        # 將 label 文字也轉換成索引方便轉換成 tensor

        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len




def create_mini_batch(samples):
    tokens_tensors1 = [s[0] for s in samples]
    segments_tensors1 = [s[1] for s in samples]

    
    # 測試集有 labels
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None
    
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
    
    return tokens_tensors1, segments_tensors1, masks_tensors1, label_ids



def get_predictions(model, dataloader, f1_valSet = False, f1_thres = 0):
    predictions = []
    single_guess = 1
    correct = 0
    fun_correct = 0
    total = 0
    y_pred = []
    y_true = []
    pred_label = []
    
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
            labels = data[3]

            for i in range(pred1.size()[0]):
                y_true.append(labels[i].item())
                y_pred.append(logits1[i][1].item())
                pred_label.append(pred1[i].item())

    
    def f1_cut(row, threshold):
        if row["y_pred"] > threshold:
            return 1
            
        else:
            return 0

    d = {'y_true': y_true, 'pred_label': pred_label, 'y_pred': y_pred}
    df = pd.DataFrame(data=d)
    
    p = metrics.precision_score(df["y_true"], df["pred_label"], average='binary')
    r = metrics.recall_score(df["y_true"], df["pred_label"], average='binary')
    
    
    if f1_valSet:
        df_f1 = df.copy()
        df_f1['pred_label_f1'] = df_f1.apply(f1_cut, threshold = f1_thres, axis=1)

        f1 = metrics.f1_score(df_f1["y_true"], df_f1["pred_label_f1"], average='binary')
#         print('precision: ', metrics.precision_score(df_f1["y_true"], df_f1["pred_label_f1"]))
#         print('recall: ', metrics.recall_score(df_f1["y_true"], df_f1["pred_label_f1"]))
#         print('f1 ', f1)
        
    else:
        #find f1 threshold
        df_f1 = df.sample(n = int(len(df)*0.02),replace=False)
    
        precisions, recalls, threshold_pr = metrics.precision_recall_curve(df_f1["y_true"], df_f1["y_pred"])

        f1_res = [-1, -1, -1, -1]
        for i in range(len(precisions)):
            devisor = (precisions[i]+recalls[i])
            if devisor ==0:    #devide by zero
                temp_f1 = 0
            else:
                temp_f1 = 2*precisions[i]*recalls[i] / (precisions[i]+recalls[i])
            if(temp_f1 > f1_res[3]):
                f1_res[0] = threshold_pr[i]
                f1_res[1] = precisions[i]
                f1_res[2] = recalls[i]
                f1_res[3] = temp_f1
#         print("threshold:", f1_res[0], "  F1: ", f1_res[3])

        df_f1 = df.drop(df_f1.index)
        df_f1['pred_label_f1'] = df_f1.apply(f1_cut, threshold = f1_res[0], axis=1)
        f1 = metrics.f1_score(df_f1["y_true"], df_f1["pred_label_f1"], average='binary')

    
    df.sort_values(by = 'y_pred', ascending=False, inplace=True)
    correct = (df['y_true'] == df['pred_label']).sum()
    total = len(df)
    acc = correct/total
    
    fpr, tpr, threshold = metrics.roc_curve(df["y_true"], df["y_pred"])
    roc_auc = metrics.auc(fpr, tpr)
    
    print('Accuracy:', acc)
    print('ROC AUC:' , roc_auc)
    print('F1:', f1)
    return predictions, acc, roc_auc, correct, total, f1



def get_val_f1(model, dataloader):
    predictions = []
    single_guess = 1
    correct = 0
    fun_correct = 0
    total = 0
    y_pred = []
    y_true = []
    pred_label = []
    
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
            labels = data[3]

            for i in range(pred1.size()[0]):
                y_true.append(labels[i].item())
                y_pred.append(logits1[i][1].item())
                pred_label.append(pred1[i].item())

    d = {'y_true': y_true, 'pred_label': pred_label, 'y_pred': y_pred}
    df = pd.DataFrame(data=d)
    df.sort_values(by = 'y_pred', ascending=False, inplace=True)
    precisions, recalls, threshold_pr = metrics.precision_recall_curve(df["y_true"], df["y_pred"])
    f1_res = [-1, -1, -1, -1]
    for i in range(len(precisions)):
        temp_f1 = 2*precisions[i]*recalls[i] / (precisions[i]+recalls[i])
        if(temp_f1 > f1_res[3]):
            f1_res[0] = threshold_pr[i]
            f1_res[1] = precisions[i]
            f1_res[2] = recalls[i]
            f1_res[3] = temp_f1
#     print("get F1 threshold: ", "threshold:", f1_res[0], "  F1: ", f1_res[3])

    return f1_res[0]




# 初始化一個每次回傳 64 個訓練樣本的 DataLoader
# 利用 `collate_fn` 將 list of samples 合併成一個 mini-batch 是關鍵
BATCH_SIZE = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
minus_num = 20


model_state_dict = torch.load(args.model_path)
model_fine = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS, state_dict=model_state_dict)
model_fine.eval()
model_fine = model_fine.to(device)


print('Shwartz random', '-'*20)
path = "../../data/evaluate/lexical_entailment/task1/Shwartz_2016/dataset_rnd/test.tsv"
testset = HeirachicalDataset(tokenizer=tokenizer)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch)
path = "../../data/evaluate/lexical_entailment/task1/Shwartz_2016/dataset_rnd/val.tsv"
valset = HeirachicalDataset(tokenizer=tokenizer)
valloader = DataLoader(valset, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch, shuffle = False)


f1_threshold = get_val_f1(model_fine, valloader)
_, acc, roc_auc, correct, total, f1 = get_predictions(model_fine, testloader, f1_valSet = True, f1_thres = f1_threshold)
torch.cuda.empty_cache() 
del model_fine
print('')

model_state_dict = torch.load(args.model_path)
model_fine = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS, state_dict=model_state_dict)
model_fine.eval()
model_fine = model_fine.to(device)


print('kotlerman', '-'*minus_num)
path = "../../data/evaluate/lexical_entailment/task1/kotlerman2010/data.tsv"
testset = HeirachicalDataset(tokenizer=tokenizer)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch)
_, acc, roc_auc, correct, total, f1 = get_predictions(model_fine, testloader)
print('')

print('Baroni', '-'*minus_num)
path = "../../data/evaluate/lexical_entailment/task1/baroni2012/data.tsv"
testset = HeirachicalDataset(tokenizer=tokenizer)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch)
_, acc, roc_auc, correct, total, f1 = get_predictions(model_fine, testloader)
print('')


print('bless', '-'*minus_num)
path = "../../data/evaluate/lexical_entailment/task1/bless2011/data.tsv"
testset = HeirachicalDataset(tokenizer=tokenizer)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch)
_, acc, roc_auc, correct, total, f1 = get_predictions(model_fine, testloader)
print('')


print('Levy', '-'*minus_num)
path = "../../data/evaluate/lexical_entailment/task1/levy2014/data.tsv"
testset = HeirachicalDataset(tokenizer=tokenizer)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch)
_, acc, roc_auc, correct, total, f1 = get_predictions(model_fine, testloader)
print('')