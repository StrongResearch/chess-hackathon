import torch
import torch.nn as nn
import math
from collections import OrderedDict

PGN_CHARS = " #+-./0123456789:=BKLNOQRabcdefghx{}*"

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
        self.vocab = PGN_CHARS
        self.device = device
        self.embedder = nn.Embedding(len(self.vocab), embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        enc_params = {"embed_dim": embed_dim, "nhead": nhead, "head_dim": head_dim, "ff_dim": ff_dim, "dropout":  dropout, "causal": causal, "norm_first": norm_first, "ghost": ghost, "device": device}
        layers = OrderedDict([(f"EncoderLayer{i}", TransformerEncoderBlock(**enc_params)) for i in range(nlayers)])
        self.encoder = nn.Sequential(layers)
        self.decoder = nn.Linear(embed_dim, len(self.vocab))
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embedder.weight, -initrange, initrange)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, pgn):
        return [self.vocab.index(c) for c in pgn]
    
    def decode(self, tokens):
        return [self.vocab[t] for t in tokens]
    
    def collate(self, batch, truncate_to=1_000):
        seq_lens = torch.tensor([len(seq) for seq in batch])
        max_seq_len = min(truncate_to, seq_lens.max())
        pad_lens = torch.clamp(max_seq_len - seq_lens, min=0)
        seqs = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)[:,:truncate_to]
        pad_from = max_seq_len - pad_lens
        pad_mask = (pad_from.unsqueeze(1) <= torch.arange(seqs.shape[1]))
        return seqs, pad_mask

    def forward(self, pgn_batch): # pgn_batch: list of pgn strings of varying length
        # encode and batch pgns, truncating and padding
        encoded_pgns = [torch.tensor(self.encode(pgn)) for pgn in pgn_batch]
        batch, pad_mask = self.collate(encoded_pgns)
        # Autoregressive modelling - targets are inputs shifted one to the left.
        inputs = batch[:, :-1].to(self.device)
        targets = batch[:, 1:].to(self.device)
        target_pad_mask = pad_mask[:, 1:].to(self.device)
        # run the inputs forward through the model
        inputs = self.embedder(inputs) # (batch_size, seq_len, embed_dim)
        inputs = self.pos_encoder(inputs) # (batch_size, seq_len, embed_dim)
        inputs = self.encoder(inputs) # (batch, token, embed)
        logits = self.decoder(inputs) # (batch, token, vocab)
        # return logits, targets, and target_pad_mask
        return logits, targets, target_pad_mask

    def score(self, pgn, move):
        '''
        pgn: string e.g. "1.e4 a6 2.Bc4 "
        move: string e.g. "a5 "
        '''
        # encode single pgn and proposed move
        encoded_pgn = self.encode(pgn)
        encoded_move = self.encode(move)
        inputs = torch.tensor(encoded_pgn + encoded_move).unsqueeze(0)
        # forward through the model
        inputs = self.embedder(inputs) # (batch_size, seq_len, embed_dim)
        inputs = self.pos_encoder(inputs) # (batch_size, seq_len, embed_dim)
        inputs = self.encoder(inputs) # (batch, token, embed)
        logits = self.decoder(inputs) # (batch, token, vocab)
        logits = logits[0] # batch size of 1 for scoring
        # decode probability for proposed move
        char_probabilities = []
        input_idxs_to_query = range(len(encoded_pgn) - 1, inputs.shape[1] - 1)
        for move_char_idx, inputs_idx in enumerate(input_idxs_to_query):
            move_char = encoded_move[move_char_idx]
            char_prob = softmax(logits[inputs_idx].detach())[move_char]
            char_probabilities.append(char_prob.item())
        # return the mean (?) probability for characters in the sequence
        return torch.tensor(char_probabilities).mean().item()
