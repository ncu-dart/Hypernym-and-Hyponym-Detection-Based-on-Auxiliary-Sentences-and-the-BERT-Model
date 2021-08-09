import pandas as pd
from tqdm import tqdm
import random
import torch


def make_NegativeDataset():
    neg_list = []
    pos_df  = pd.read_csv('../../data/train/task12_clean_heirachical.txt', sep="\t", header=None)
    len_pos_df = len(pos_df)
    for idx in tqdm(range(len(pos_df))):
        text_a, _ = pos_df.iloc[idx, :].values
        while True:
            idx_negative = random.randint(0, len_pos_df-1)
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


for i in range(20):
    print('epoch', i)
    neg_list = make_NegativeDataset()
    with open('../../data/train/negative_data/negative_sample' + str(i) + '.txt', 'w') as f:
        for item in neg_list:
            f.write("%s\n" % item)






