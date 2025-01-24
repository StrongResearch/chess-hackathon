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

class RawEvalDataset(Dataset):
    def __init__(self, source_dir, file_prefix="EVALS", file_suffix=".store", extras=False, symmetric=True):
        super().__init__()
        self.source_dir = source_dir
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.extras = extras
        self.symmetric = symmetric
        self.PIECE_CHARS = "♔♕♖♗♘♙⭘♟♞♝♜♛♚"
        # internal registers
        self.fens = []
        self.pgn_sets = []
        self.sans = []
        self.counts = []
        self.scores = []
        # sorted to ensure consistency
        data_files = sorted([f for f in os.listdir(source_dir) if f.startswith(self.file_prefix) and f.endswith(self.file_suffix)])
        for file in data_files:
            data = torch.load(os.path.join(source_dir, file))
            # sorted to ensure consistency
            _, fen_data_maps = zip(*sorted(data.items(), key = lambda t: t[0]))
            batch_fens, batch_pgn_sets, batch_sans, batch_counts, batch_scores = [], [], [], [], []
            for fen_map in fen_data_maps:
                batch_fens.append(fen_map["fen"])
                batch_scores.append(fen_map["score"])
                # maybe also load extras
                if self.extras:
                    batch_pgn_sets.append(fen_map["pgns"])
                    batch_sans.append(fen_map["san"])
                    batch_counts.append(fen_map["count"])
            # append batch to internal registers
            self.fens += batch_fens
            self.pgn_sets += batch_pgn_sets
            self.sans += batch_sans
            self.counts += batch_counts
            self.scores += batch_scores
        # purge working data from memory
        del data
        del batch_fens
        del batch_pgn_sets
        del batch_sans
        del batch_counts
        del batch_scores
        # summarize
        self.len = len(self.fens)
    
    def encode_board(self, board: Board) -> torch.tensor:
        # If symmetric, then flip the board to the perspective of the player to move and re-code pieces to be "mine" and "theirs"
        # If board.turn = True then the board thinks it is white's turn, which means this is a potential move being 
        # contemplated by black. Therefore we flip the board to black's perspective.
        # If board.turn = False then the board thinks it is black's turn, which means this is a potential move being
        # conteplated by white. Therefore we leave the board the way it is.
        step = 1 - 2 * board.turn if self.symmetric else 1
        unicode = board.unicode().replace(' ','').replace('\n','')[::step]
        return torch.tensor([self.PIECE_CHARS[::step].index(c) for c in unicode], dtype=torch.uint8).reshape(8,8)

    def __len__(self):
        return self.len
            
    def __getitem__(self, idx):
        fen = self.fens[idx]
        score = self.scores[idx]
        board_tensor = self.encode_board(Board(fen))
        score_tensor = torch.tensor(score, dtype=torch.float16)
        if self.extras:
            pgns = self.pgn_sets[idx]
            san = self.sans[idx]
            count = self.counts[idx]
            return board_tensor, score_tensor, fen, pgns, san, count
        else:
            return board_tensor, score_tensor
