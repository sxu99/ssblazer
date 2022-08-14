import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

nucleobase_mapping = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "a": [1, 0, 0, 0],
    "c": [0, 1, 0, 0],
    "g": [0, 0, 1, 0],
    "t": [0, 0, 0, 1]
}

def one_hot_encode(seq):
    encoded_seq = []
    for c in seq:
        encoded_seq.append(nucleobase_mapping[c])
    return encoded_seq

def collate_fn(data):
    seq_batch = []

    for i in range(len(data)):
        dic = data[i]
        seq_batch.append(one_hot_encode(dic['seq']))
    seq_batch = torch.Tensor(seq_batch)

    res = {}
    res['seq'] = seq_batch
    return res

class DatasetFromStr(Dataset):
    def __init__(self, seq):
        start_pos=0
        end_pos=151
        seq_clip_list=[]
        while(end_pos<=len(seq)):
            seq_clip=seq[start_pos:end_pos]
            seq_clip_list.append(seq_clip)
            start_pos+=1
            end_pos+=1

        seq_df=pd.DataFrame(data=seq_clip_list,columns=['SEQ'])

        self.data = seq_df

    def __getitem__(self, index):
        seq = self.data.iloc[index]['SEQ']
        sample = {'seq': seq}
        return sample

    def __len__(self):
        return len(self.data.index)