import os
import numpy as np
import torch
from torch.utils.data import Dataset
from itertools import accumulate
from h5py import File as h5pyFile

class PGN_HDF_Dataset(Dataset):
    def __init__(self, source_dir=None, meta=False):
        self.source_dir = source_dir
        self.meta = meta
        with open(os.path.join(self.source_dir, "inventory.txt"), "r") as file:
            self.inventory = file.readlines()
        sizes, self.filenames = zip(*[i.split() for i in self.inventory[1:]])
        self.sizes = [int(s) for s in sizes]
        self.len = sum(self.sizes)
        self.breaks = np.array(list(accumulate(self.sizes)))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        hdf_idx = (self.breaks > idx).argmax().item()
        pgn_idx = idx - sum(self.sizes[:hdf_idx])
        hdf_path = os.path.join(self.source_dir, self.filenames[hdf_idx])
        with h5pyFile(hdf_path, 'r') as hf:
            pgn = hf["pgn"][pgn_idx].decode('utf-8')
            if self.meta:
                meta = hf["meta"][pgn_idx].decode('utf-8')
        if self.meta:
            return pgn, meta
        else:
            return pgn
    
class EVAL_HDF_Dataset(Dataset):
    def __init__(self, source_dir):
        super().__init__()
        self.source_dir = source_dir
        with open(os.path.join(self.source_dir, "inventory.txt"), "r") as file:
            self.inventory = file.readlines()
        sizes, self.filenames = zip(*[i.split() for i in self.inventory[1:]])
        self.sizes = [int(s) for s in sizes]
        self.len = sum(self.sizes)
        self.breaks = np.array(list(accumulate(self.sizes)))

    def __len__(self):
        return self.len
            
    def __getitem__(self, idx):
        hdf_idx = (self.breaks > idx).argmax().item()
        board_idx = idx - sum(self.sizes[:hdf_idx])
        hdf_path = os.path.join(self.source_dir, self.filenames[hdf_idx])
        with h5pyFile(hdf_path, 'r') as hf:
            board = hf["boards"][board_idx]
            score = hf["scores"][board_idx]
        board = torch.from_numpy(board)
        score = torch.tensor(score)     
        return board, score