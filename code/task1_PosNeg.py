from argparse import ArgumentParser
from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained_bert import BertForSequenceClassification
import numpy as np
import pandas as pd
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

parser = ArgumentParser()
parser.add_argument("-dataset", help="path to dataset", dest="dataset", default = '../../data/train/test_train.txt')
parser.add_argument("-label", help="training set with label or not", dest="labeled", default = 'False')
parser.add_argument("-b", help="batch size, default 64", dest="b", default=64)
parser.add_argument("-epoch", help="number of epochs, default 19", dest="epoch", default=19)
parser.add_argument("-lr", help="learning rate, default 1e-5", dest="lr", default=1e-5)
parser.add_argument("-model", help="model name", dest="model_name", default='output/task1_PosNeg_')

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
#         assert mode in ["train", "test"]  # 一般訓練你會需要 dev set
#         self.mode = mode
        # 大數據你會需要用 iterator=True
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

        len_a = len(word_pieces)
        
        y_pred = ""
        
        if str2bool(args.labeled) == True:
            if label == True:
                if self.mode == "positive":
                    y_pred = "positive"
                    label_tensor = torch.tensor(1)
                else:
                    y_pred = "positive"
                    label_tensor = torch.tensor(0)
                
            else:
                if self.mode == "positive":
                    y_pred = "positive"
                    label_tensor = torch.tensor(0)
                else:
                    y_pred = "positive"
                    label_tensor = torch.tensor(1)
        else:
            if self.mode == "positive":
                y_pred = "positive"
                label_tensor = torch.tensor(1)
            else:
                y_pred = "negative"
                label_tensor = torch.tensor(0)

        token_y_pred = self.tokenizer.tokenize(y_pred)
        word_pieces += ( token_y_pred + ["[SEP]"] )
        
        len_b = len(word_pieces) - len_a

        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a + [1] * len_b, 
                                        dtype=torch.long)
       
        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len
    
    
def make_NegativeDataset():
    neg_list = []
    pos_df  = pd.read_csv('task12_clean_heirachical.txt', sep="\t", header=None)
    self.len = len(self.df)
    for idx in tqdm_notebook(range(len_pos_df)):
        text_a, _ = pos_df.iloc[idx, :].values

        while True:
            idx_negative = random.randint(0, self.len-1)
            choice0, choice1 = pos_df.iloc[idx_negative, :].values
            if torch.randint(0, 1, (1, 1)).item() == 0:
                text_b = choice0
            else:
                text_b = choice1
            #check if text_a, text_b already in df 
            select_df = pos_df[pos_df[0] == text_a]
            if len(select_df[select_df[1] == text_b]):
                continue
            else:
                neg_list.append(text_a + "\t" + text_b)
                break

    return neg_list


class NegativeDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, mode, epoch, tokenizer):
        self.epoch = epoch
        self.df  = pd.read_csv('../../data/train/negative_data/negative_sample' + str(self.epoch) + '.txt', sep="\t", header=None)
        self.len = len(self.df)
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer
        self.mode = mode
    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
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
        
        word_pieces += (tokens_a + tokens_middle + tokens_b + tokens_tail + ["[SEP]"])
        len_a = len(word_pieces)
        
        y_pred = ""
        
        if self.mode == "positive":
            y_pred = "positive"
            label_tensor = torch.tensor(0)
        else:
            y_pred = "negative"
            label_tensor = torch.tensor(1)

        token_y_pred = self.tokenizer.tokenize(y_pred)
        word_pieces += ( token_y_pred + ["[SEP]"] )
        
        len_b = len(word_pieces) - len_a

        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a + [1] * len_b, 
                                        dtype=torch.long)
        

        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len
    
trainset_normal_positive = HeirachicalDataset(mode = "positive", tokenizer=tokenizer)
trainset_normal_negative = HeirachicalDataset(mode = "negative", tokenizer=tokenizer)

trainset_abnormal_positive = NegativeDataset(mode = "positive", epoch = 0,  tokenizer=tokenizer)
trainset_abnormal_negative = NegativeDataset(mode = "negative", epoch = 0,  tokenizer=tokenizer)

if str2bool(args.labeled):
    trainset = trainset_normal_positive + trainset_normal_negative
else:
    trainset = trainset_normal_positive + trainset_normal_negative + trainset_abnormal_positive + trainset_abnormal_negative

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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = model.to(device)

model.train()

# 使用 Adam Optim 更新整個分類模型的參數
optimizer = torch.optim.Adam(model.parameters(), args.lr)
epoch_loss = []

EPOCHS = args.epoch
print('training for ' , EPOCHS, 'epochs, learning rate=', args.lr, ', batch size=', args.b)
for epochs in range(EPOCHS):
    trainset_abnormal_positive = NegativeDataset(mode = "positive", epoch = epochs,  tokenizer=tokenizer)
    trainset_abnormal_negative = NegativeDataset(mode = "negative", epoch = epochs,  tokenizer=tokenizer)
    trainset = trainset_normal_positive + trainset_normal_negative + trainset_abnormal_positive + trainset_abnormal_negative
    trainloader = DataLoader(trainset, batch_size=args.b, 
                         collate_fn=create_mini_batch, shuffle=True)

    
    running_loss = 0.0
    trainloader = DataLoader(trainset, batch_size=args.b, 
                         collate_fn=create_mini_batch, shuffle=True)

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
    
    file_name = arg.model_name + str(epochs) + '.pth'
    
    with open(arg.model_name + '_record.txt', 'a') as f:
        f.write("%f\t%f\n"%(epochs, running_loss))
    
    epoch_loss.append(running_loss)


    torch.save(model.state_dict(), file_name)
