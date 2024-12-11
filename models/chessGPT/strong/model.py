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

def multihead_cross_attention(Q, K, V, rope_encoder=None, mask=None, ghost=None, device='cpu'):
    '''
    Accepts input of Q, K, V each with shape (batch_size, nhead, seq_len, head_dim),
    or more generally with shape (..., seq_len, head_dim).
    If passed, "mask" is assumed to be a tensor of floats of shape (seq_len, seq_len) 
    Returns attention tensor A of shape (..., seq_len, head_dim).
    '''
    head_dim = Q.shape[-1]

    if rope_encoder:
        Q = rope_encoder(Q)
        K = rope_encoder(K)

    QKT = torch.einsum('...Qe,...Ke->...QK', Q, K) / math.sqrt(head_dim)

    if isinstance(mask, torch.Tensor):
        QKT += mask

    S = softmax(QKT, dim=-1, ghost=ghost)
    A = torch.einsum('...SV,...Ve->...Se', S, V)
    return A

class MultiHeadSelfAttention(nn.Module):
    '''
    Assumes input with shape (batch_size, seq_len, embed_dim).
    If causal, causal mask is generated and applied.
    '''
    def __init__(self, embed_dim=512, nhead=8, head_dim=64, rope=True, causal=True, ghost=False, device='cpu'):
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
        self.rope_encoder = RotaryPositionalEmbeddings(head_dim) if rope else None
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.Wqkv.weight)
        nn.init.xavier_uniform_(self.Wo.weight)
        nn.init.zeros_(self.Wqkv.bias)
        nn.init.zeros_(self.Wo.bias)

    def forward(self, inputs):
        batch_size, seq_len, _embed_dim = inputs.shape
        mask = None
        if self.causal:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, dtype=torch.float, device=self.device)

        QKV = self.Wqkv(inputs)
        QKVh = QKV.reshape(batch_size, seq_len, 3, self.nhead, self.head_dim).transpose(1, 3)
        Q, K, V = [t.squeeze(2) for t in QKVh.split(1, 2)] # squeezing out the projection dimension only
        A = multihead_cross_attention(Q, K, V, self.rope_encoder, mask, self.ghost, self.device)
        A = A.transpose(1, 2).reshape(batch_size, seq_len, -1)
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
    def __init__(self, embed_dim=512, nhead=8, head_dim=64, ff_dim=2048, dropout=0.1, rope=True, causal=True, norm_first=False, ghost=False, device='cpu'):
        super().__init__()
        self.norm_first = norm_first
        mhsa_params = {"embed_dim": embed_dim, "nhead": nhead, "head_dim": head_dim, "rope": rope, "causal": causal, "ghost": ghost, "device": device}
        self.self_attention = MultiHeadSelfAttention(**mhsa_params)
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

class Model(nn.Module):
    """Transformer Model"""
    def __init__(self, nlayers=10, embed_dim=512, nhead=8, head_dim=64, ff_dim=2048, dropout=0.1, rope=True, causal=True, norm_first=False, ghost=False, device='cpu'):
        super().__init__()
        self.vocab = PGN_CHARS
        self.device = device
        self.rope = rope
        self.embedder = nn.Embedding(len(self.vocab), embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout) if rope == False else None
        encoder_params = {"embed_dim": embed_dim, "nhead": nhead, "head_dim": head_dim, "ff_dim": ff_dim, "dropout":  dropout, "rope": rope, "causal": causal, "norm_first": norm_first, "ghost": ghost, "device": device}
        layers = OrderedDict([(f"EncoderLayer{i}", TransformerEncoderBlock(**encoder_params)) for i in range(nlayers)])
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
        '''
        Truncates sequences longer than truncate_to tokens.
        Applies tail padding of sequences shorter than truncate_to tokens.
        Returns the padded sequence and also a boolean mask indicating which tokens are padding.
        '''
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
        if self.pos_encoder:
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
        if self.pos_encoder:
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


## -- Positional Encoding Strategies -- ##

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

class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    This implementation caches the embeddings for each position upto
    ``max_seq_len`` at init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10_000) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim))
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        # Outer product of theta and position index; output tensor has a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        # cache includes both the cos and sin components and so the output shape is [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``
        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the applicable cache values
        rope_cache = self.cache[:seq_len]

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples, otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)
