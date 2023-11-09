import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class DatasetFromCSV(Dataset):
    def __init__(self, csv_path, bootstrap=False):
        df = pd.read_csv(csv_path)
        df["SEQ"] = df["seq"].astype(str)
        df["LABEL"]= df["label"].astype(int)

        if(bootstrap):
            count = df['LABEL'].value_counts()
            self.class_num_list = count.tolist()
            df = df.sort_values(by=['LABEL'], ascending=True)
        self.data = df

    def __getitem__(self, index):
        label = self.data.iloc[index]["LABEL"]
        seq = self.data.iloc[index]['SEQ']
        sample = {'seq': seq, 'label': label}
        return sample

    def __len__(self):
        return len(self.data.index)
    


class DataModule(LightningDataModule):
    def __init__(self, batch_size, n_workers, dataset):
        super().__init__()
        self.dataset = dataset
        self.n_workers = n_workers if n_workers is not None else os.cpu_count()
        self.batch_size = batch_size


    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the training DataLoader."""
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: self.prepare_batch(x),
            pin_memory=True,
            num_workers=self.n_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the validation DataLoader."""
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: self.prepare_batch(x),
            pin_memory=True,
            num_workers=self.n_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the test DataLoader."""
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: self.prepare_batch(x),
            pin_memory=True,
            num_workers=self.n_workers,
            shuffle=False,
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the predict DataLoader."""
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: self.prepare_batch(x),
            pin_memory=True,
            num_workers=self.n_workers,
            shuffle=False,
        )

    def prepare_batch(self, data):
        seq_batch = []
        label_batch = []

        for i in range(len(data)):
            dic = data[i]
            seq_batch.append(self.one_hot_encode(dic["seq"]))
            label_batch.append(dic["label"])
        seq_batch = torch.Tensor(seq_batch)

        res = {}
        res["seq"] = seq_batch
        res["label"] = torch.Tensor(label_batch)
        return res

    def one_hot_encode(self, seq):
        nucleobase_mapping = {
            "A": [1, 0, 0, 0],
            "C": [0, 1, 0, 0],
            "G": [0, 0, 1, 0],
            "T": [0, 0, 0, 1],
            "a": [1, 0, 0, 0],
            "c": [0, 1, 0, 0],
            "g": [0, 0, 1, 0],
            "t": [0, 0, 0, 1],
            "N":[0,0,0,0],
            "n":[0,0,0,0]
        }
        encoded_seq = []
        for c in seq:
            encoded_seq.append(nucleobase_mapping[c])
        return encoded_seq
