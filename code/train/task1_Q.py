from argparse import ArgumentParser
from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained_bert import BertForSequenceClassification
import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument("-dataset", help="path to dataset", dest="dataset", default = '../../data/train/task12_clean_heirachical.txt')
parser.add_argument("-label", help="training set with label or not", dest="labeled", default = 'False')
parser.add_argument("-b", help="batch size, default 64", dest="b", default=64)
parser.add_argument("-epoch", help="number of epochs, default 19", dest="epoch", default=19)
parser.add_argument("-lr", help="learning rate, default 1e-5", dest="lr", default=1e-5)
parser.add_argument("-model", help="model name", dest="model_name", default='output/task1_Q_')

args = parser.parse_args()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# 加载bert的分詞器
PRETRAINED_MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)



print('load BERT pretrained model...')

NUM_LABELS = 2
model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)

# high-level 顯示此模型裡的 modules
print("""
name            module
----------------------""")
for name, module in model.named_children():
    if name == "bert":
        for n, _ in module.named_children():
            print(f"{name}:{n}")
    else:
        print("{:15} {}".format(name, module))



class HeirachicalDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, mode, tokenizer):
        self.df  = pd.read_csv(args.dataset, sep="\t", header=None)
        self.len = len(self.df)
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer
        self.mode = mode
    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        if str2bool(args.labeled) == True:
            text_a, text_b, label = self.df.iloc[idx, :].values
        else:
            text_a, text_b = self.df.iloc[idx, :].values
        
        middle_words = "and"
        tail_words = "are hierarchically related"

        # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
        word_pieces = ["[CLS]"]
        
        fake_a = ""
        tokens_a = self.tokenizer.tokenize(fake_a)
        
        try:
            tokens_a = self.tokenizer.tokenize(text_a)
        except:
            print(idx, text_a, text_b)
            exit()
        

        # 第二個句子的 BERT tokens
        tokens_b = self.tokenizer.tokenize(text_b)
        tokens_middle = self.tokenizer.tokenize(middle_words)
        tokens_tail = self.tokenizer.tokenize(tail_words)
        
        if random.randint(0, 1) == 0:
            word_pieces += (tokens_a + tokens_middle + tokens_b + tokens_tail + ["[SEP]"])
        else:
            word_pieces += (tokens_b + tokens_middle + tokens_a + tokens_tail + ["[SEP]"])

            
        len_all = len(word_pieces)

        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0]* len_all, 
                                        dtype=torch.long)
        
        # 將 label 文字也轉換成索引方便轉換成 tensor
        if str2bool(args.labeled) == True:
            if label == True:
                label_tensor = torch.tensor(1)
            else:
                label_tensor = torch.tensor(0)
        else:
            label_tensor = torch.tensor(1)
            
        return (tokens_tensor, segments_tensor, label_tensor)
        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len
    
    


class NegativeDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, tokenizer):
#         assert mode in ["train", "test"]  # 一般訓練你會需要 dev set
#         self.mode = mode
        # 大數據你會需要用 iterator=True
        self.df  = pd.read_csv(args.dataset, sep="\t", header=None)
        self.len = len(self.df)
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer

    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        text_a, _ = self.df.iloc[idx, :].values
        
        rand_Num_Index = 0

        while True:
            idx_negative = random.randint(0, self.len -1)
            choice0, choice1 = self.df.iloc[idx_negative, :].values
            if torch.randint(0, 1, (1, 1)).item() == 0:
                text_b = choice0
            else:
                text_b = choice1
            #check if text_a, text_b already in df 
            select_df = self.df[self.df[0] == text_a]
            if len(select_df[select_df[1] == text_b]):
                rand_Num_Index += 1
                continue
            else:
                break

        
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


        word_pieces += (tokens_a + tokens_middle + tokens_b + tokens_tail + ["[SEP]"])
        len_all = len(word_pieces)

        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0]* len_all, 
                                        dtype=torch.long)
        
        
        # 將 label 文字也轉換成索引方便轉換成 tensor
        label_tensor = torch.tensor(0)

        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len
    
    

trainset_normal = HeirachicalDataset(mode= 'normal', tokenizer=tokenizer)
trainset_reverse = HeirachicalDataset(mode= 'reverse', tokenizer=tokenizer)

trainset_negative1 = NegativeDataset(tokenizer=tokenizer)
trainset_negative2 = NegativeDataset(tokenizer=tokenizer)

if str2bool(args.labeled):
    trainset = trainset_normal + trainset_normal
else:
    trainset = trainset_normal + trainset_normal + trainset_negative1 + trainset_negative1


def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    
    # 測試集有 labels
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None
    
    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, 
                                    batch_first=True)
    
    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids


# 初始化一個每次回傳 64 個訓練樣本的 DataLoader
# 利用 `collate_fn` 將 list of samples 合併成一個 mini-batch 是關鍵
trainloader = DataLoader(trainset, batch_size=args.b, 
                         collate_fn=create_mini_batch, shuffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = model.to(device)



model.train()

# 使用 Adam Optim 更新整個分類模型的參數
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
epoch_loss = []

EPOCHS = args.epoch
print('training for ' , EPOCHS, 'epochs, learning rate=', args.lr, ', batch size=', args.b)

for epochs in range(EPOCHS):

    running_loss = 0.0
    print('epoch', epochs)
    for data in tqdm(trainloader):
        
        tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]

        # 將參數梯度歸零
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(input_ids=tokens_tensors, 
                        token_type_ids=segments_tensors, 
                        attention_mask=masks_tensors, 
                        labels=labels)

#         print(outputs)
        loss = outputs
        # backward
        loss.backward()
        optimizer.step()


        # 紀錄當前 batch loss
        running_loss += loss.item()
    file_name = args.model_name + str(epochs) + '.pth'
    torch.save(model.state_dict(), file_name)
    
    with open(args.model_name + 'loss_record.txt', 'a') as f:
        f.write("%d\t%f\n"%(epochs, running_loss))
    
    epoch_loss.append(running_loss)

