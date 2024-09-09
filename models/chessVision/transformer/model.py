import io
import torch
import torch.nn as nn
import math
import chess.pgn
from chess import Board
from collections import OrderedDict

PIECE_CHARS = "♔♕♖♗♘♙⭘♟♞♝♜♛♚"

def encode_board(board: Board) -> np.array:
    # String-encode the board.
    # If board.turn = 1 then it is now white's turn which means this is a potential move
    # being contemplated by black, and therefore we reverse the char order to rotate the board
    # for black's perspective
    # If board.turn = 0 then it is now black's turn which means this is a potential move
    # being contemplated by white, and therefore we leave the char order as white's perspective.
    # Also reverse PIECE_CHARS indexing order if black's turn to reflect "my" and "opponent" pieces.
    step = 1 - 2 * board.turn
    unicode = board.unicode().replace(' ','').replace('\n','')[::step]
    return np.array([PIECE_CHARS[::step].index(c) for c in unicode], dtype=int).reshape(8,8)

def softmax(x, dim=-1, temp=1, ghost=None):
    z = torch.exp((x - torch.max(x, dim=dim, keepdim=True).values) / temp)
    z_sum = z.sum(dim=dim, keepdim=True)
    if isinstance(ghost, torch.Tensor):
        z_sum += ghost.view(1, -1, 1, 1)
    return z / z_sum

def multihead_cross_attention(Q, K, V, causal=True, ghost=None, device='cpu'):
    '''
    Accepts input of Q, K, V each with shape (batch_size, nhead, seq_len, head_dim),
    or more generally with shape (..., seq_len, head_dim).
    If causal, causal mask is generated and applied.
    Returns attention tensor A of shape (..., seq_len, head_dim).
    '''
    _batch_size, _nhead, seq_len, head_dim = Q.shape
    QKT = torch.einsum('...Qe,...Ke->...QK', Q, K) / math.sqrt(head_dim)
    if causal:
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, dtype=torch.float, device=device)
        mask = mask.view(1, 1, seq_len, seq_len)
        QKT += mask
    S = softmax(QKT, dim=-1, ghost=ghost)
    A = torch.einsum('...SV,...Ve->...Se', S, V)
    return A

class MultiHeadSelfAttention(nn.Module):
    '''
    Assumes input with shape (batch_size, seq_len, embed_dim).
    If causal, causal mask is generated and applied.
    '''
    def __init__(self, embed_dim=512, nhead=8, head_dim=64, causal=True, ghost=False, device='cpu'):
        super().__init__()
        self.nhead = nhead
        self.head_dim = head_dim
        self.causal = causal
        self.ghost = ghost
        self.device = device
        # ghost is one learnable param per attention head
        self.ghost = nn.parameter.Parameter(data=torch.zeros(nhead)) if ghost else None
        self.Wqkv = nn.Linear(embed_dim, 3 * nhead * head_dim)
        self.Wo = nn.Linear(nhead * head_dim, embed_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.Wqkv.weight)
        nn.init.xavier_uniform_(self.Wo.weight)
        nn.init.zeros_(self.Wqkv.bias)
        nn.init.zeros_(self.Wo.bias)

    def forward(self, inputs):
        batch_size, seq_len, embed_dim = inputs.shape
        QKV = self.Wqkv(inputs)
        QKVh = QKV.reshape(batch_size, seq_len, 3, self.nhead, self.head_dim).transpose(1, 3)
        Q, K, V = [t.squeeze(2) for t in QKVh.split(1, 2)] # squeezing out the projection dimension only
        A = multihead_cross_attention(Q, K, V, causal=self.causal, ghost=self.ghost, device=self.device).transpose(1, 2).reshape(batch_size, seq_len, -1)
        outputs = self.Wo(A)
        return outputs

class FeedForward(nn.Module):
    def __init__(self, embed_dim=512, ff_dim=2048):
        super().__init__()
        self.lin1 = nn.Linear(embed_dim, ff_dim, bias=True)
        self.lin2 = nn.Linear(ff_dim, embed_dim, bias=True)
        self.act = nn.LeakyReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.zeros_(self.lin2.bias)
        
    def forward(self, inputs):
        return self.lin2(self.act(self.lin1(inputs)))

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=512, nhead=8, head_dim=64, ff_dim=2048, dropout=0.1, causal=True, norm_first=False, ghost=False, device='cpu'):
        super().__init__()
        self.norm_first = norm_first
        self.self_attention = MultiHeadSelfAttention(embed_dim, nhead, head_dim, causal, ghost, device)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feedforward = FeedForward(embed_dim, ff_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, inputs):
        if self.norm_first:
            inputs = inputs + self.dropout(self.self_attention(self.norm1(inputs)))
            inputs = inputs + self.dropout(self.feedforward(self.norm2(inputs)))
        else:
            inputs = self.norm1(inputs + self.dropout(self.self_attention(inputs)))
            inputs = self.norm2(inputs + self.dropout(self.feedforward(inputs)))
        return inputs

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Model(nn.Module):
    """Transformer Model"""
    def __init__(self, nlayers=10, embed_dim=512, nhead=8, head_dim=64, ff_dim=2048, dropout=0.1, causal=True, norm_first=False, ghost=False, device='cpu'):
        super().__init__()
        self.vocab = PIECE_CHARS
        self.device = device
        self.embedder = nn.Embedding(len(self.vocab), embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        enc_params = {"embed_dim": embed_dim, "nhead": nhead, "head_dim": head_dim, "ff_dim": ff_dim, "dropout":  dropout, "causal": causal, "norm_first": norm_first, "ghost": ghost, "device": device}
        layers = OrderedDict([(f"EncoderLayer{i}", TransformerEncoderBlock(**enc_params)) for i in range(nlayers)])
        self.encoder = nn.Sequential(layers)
        self.reducer = nn.Linear(embed_dim, 1)
        self.decoder = nn.Linear(64, 1)
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.embedder.weight, -1.0, 1.0)
        nn.init.xavier_uniform_(self.reducer.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.reducer.bias)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, boards): # boards: tensor of boards (batch, 8, 8)
        boards = boards.flatten(1) # (batch, 64)
        boards = self.embedder(boards) # (batch, 64, embed)
        boards = self.pos_encoder(boards) # (batch, 64, embed)
        boards = self.encoder(boards) # (batch, 64, embed)
        boards = self.reducer(boards).squeeze() # (batch, 64)
        boards = self.decoder(boards).squeeze() # (batch)
        return boards.squeeze()

    def score(self, pgn, move):
        '''
        pgn: string e.g. "1.e4 a6 2.Bc4 "
        move: string e.g. "a5 "
        '''
        # init a game and board
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = Board()
        # catch board up on game to present
        for past_move in list(game.mainline_moves()):
            board.push(past_move)
        # push the move to score
        board.push_san(move)
        # convert to tensor, unsqueezing a dummy batch dimension
        board_tensor = torch.tensor(encode_board(board)).unsqueeze(0)
        return self.forward(board_tensor).item()
