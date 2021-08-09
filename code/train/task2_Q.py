from argparse import ArgumentParser
from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained_bert import BertForSequenceClassification
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("-dataset", help="path to dataset", dest="dataset", default = 'data/train/test_train.txt')
parser.add_argument("-b", help="batch size, default 64", dest="b", default=64)
parser.add_argument("-epoch", help="number of epochs, default 7", dest="epoch", default=7)
parser.add_argument("-lr", help="learning rate, default 1e-5", dest="lr", default=1e-5)
parser.add_argument("-model", help="model name", dest="model_name", default='output/task2_Q_')

args = parser.parse_args()

# 加載bert的分詞器
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


# ## simple

# In[4]:


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
        text_a, text_b = self.df.iloc[idx, :].values
        # 將 label 文字也轉換成索引方便轉換成 tensor
        label_tensor = torch.tensor(1)
        
        middle_Words = "is a type of"

        # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
        word_pieces = ["[CLS]"]
        
        fake_a = ""
        tokens_a = self.tokenizer.tokenize(fake_a)
        
        try:
            tokens_a = self.tokenizer.tokenize(text_a)
        except:
            print(idx, text_a, text_b)
            exit()
        
        tokens_middle = self.tokenizer.tokenize(middle_Words)
        

        # 第二個句子的 BERT tokens
        tokens_b = self.tokenizer.tokenize(text_b)
        

        if self.mode == "reverse":
            answer = "negative"
            ans_token = self.tokenizer.tokenize(answer) 
            word_pieces += (tokens_b + tokens_middle + tokens_a + ["[SEP]"])
            label_tensor = torch.tensor(0)
        else:
            answer = "positive"
            ans_token = self.tokenizer.tokenize(answer) 
            word_pieces += tokens_a + tokens_middle + tokens_b + ["[SEP]"]
            label_tensor = torch.tensor(1)
            
        len_all = len(word_pieces)

        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0]* len_all, 
                                        dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len
    
    
trainset_normal = HeirachicalDataset(mode="normal", tokenizer=tokenizer)
trainset_reversed1 = HeirachicalDataset(mode = "reverse", tokenizer=tokenizer)
trainset_reversed2 = HeirachicalDataset(mode = "reverse", tokenizer=tokenizer)

trainset = trainset_normal + trainset_reversed1 + trainset_normal + trainset_reversed2


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


EPOCHS = args.epoch
print('training for ' , EPOCHS, 'epochs, learning rate=', args.lr, ', batch size=', args.b)
for epoch in range(EPOCHS):
    print('epoch', epoch)
    running_loss = 0.0
    for data in tqdm(trainloader):
        
        tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]

        # 將參數梯度歸零
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(input_ids=tokens_tensors, 
                        token_type_ids=segments_tensors, 
                        attention_mask=masks_tensors, 
                        labels=labels)

        loss = outputs
        # backward
        loss.backward()
        optimizer.step()


        # 紀錄當前 batch loss
        running_loss += loss.item()
    file_name = args.model_name + str(epoch) + '.pth'
    torch.save(model.state_dict(), file_name)
    
    with open(args.model_name + 'record.txt', 'a') as f:
        f.write("%d\t%f\n"%(epoch, running_loss))
