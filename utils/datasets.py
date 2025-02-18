import os
import numpy as np
import torch
from torch.utils.data import Dataset
from itertools import accumulate
from h5py import File as h5pyFile
from chess import Board

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

class RandomGames_EVAL_HDF_Dataset(Dataset):
    def __init__(self, source_dir, symmetric_board=True, extras=False):
        super().__init__()
        self.source_dir = source_dir
        self.symmetric_board = symmetric_board
        self.extras = extras
        self.PIECE_CHARS = "♔♕♖♗♘♙⭘♟♞♝♜♛♚"
        with open(os.path.join(self.source_dir, "inventory.txt"), "r") as file:
            self.inventory = file.readlines()
        sizes, self.filenames = zip(*[i.split() for i in self.inventory[1:]])
        self.sizes = [int(s) for s in sizes]
        self.len = sum(self.sizes)
        self.breaks = np.array(list(accumulate(self.sizes)))

    def encode_board(self, board: Board) -> torch.tensor:
        # If symmetric_board, then flip the board to the perspective of the player to move and re-code pieces to be "mine" and "theirs"
        # If board.turn = True then the board thinks it is white's turn, which means this is a potential move being 
        # contemplated by black. Therefore we flip the board to black's perspective.
        # If board.turn = False then the board thinks it is black's turn, which means this is a potential move being
        # conteplated by white. Therefore we leave the board the way it is.
        step = 1 - 2 * board.turn if self.symmetric_board else 1
        unicode = board.unicode().replace(' ','').replace('\n','')[::step]
        return torch.tensor([self.PIECE_CHARS[::step].index(c) for c in unicode], dtype=torch.int).reshape(8,8)

    def __len__(self):
        return self.len
            
    def __getitem__(self, idx):
        hdf_idx = (self.breaks > idx).argmax().item()
        board_idx = idx - sum(self.sizes[:hdf_idx])
        hdf_path = os.path.join(self.source_dir, self.filenames[hdf_idx])
        with h5pyFile(hdf_path, 'r') as hf:
            fen = hf["fens"][board_idx]
            score = hf["scores"][board_idx]
            if self.extras:
                pgn_set = hf["pgn_sets"][board_idx]
                san = hf["sans"][board_idx]
                count = hf["counts"][board_idx]

        board = self.encode_board(Board(fen.decode()))
        score = torch.tensor(score, dtype=torch.float32)
        if self.symmetric_board:
            score *= -1
        if self.extras:
            count = torch.tensor(count, dtype=torch.int)
            return board, score, fen, pgn_set, san, count
        else:
            return board, score
