import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(2024) #set seed
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#### Hyperparamters ####
block_size = 256 # length for predictions
epochs = 3000
learning_rate = 3e-4
dim_embd = 384
n_heads = 6
n_blocks = 6
dropout = .1
#########################

#### Helper Classes ####
class HeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(dim_embd, dim_embd//n_heads, bias=False)
        self.query = nn.Linear(dim_embd, dim_embd//n_heads, bias=False)
        self.value = nn.Linear(dim_embd, dim_embd//n_heads, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) 
        q = self.query(x) 
        wei = q @ k.transpose(-2, -1)* C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v
    
class HeadsAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([HeadAttention() for _ in range(n_heads)])
        self.proj = nn.Linear(dim_embd, dim_embd) 
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)
    
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_embd, 4*dim_embd),
            nn.ReLU(),
            nn.Linear(4*dim_embd, dim_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = HeadsAttention()
        self.fforward = FeedForward()
        self.norm1 = nn.LayerNorm(dim_embd)
        self.norm2 = nn.LayerNorm(dim_embd)
        
    def forward(self, x):
        x = x + self.heads(self.norm1(x)) 
        x = x + self.fforward(self.norm2(x))
        return x
#########################

#### Model #####
class ShakePT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, dim_embd)
        self.position_embedding_table = nn.Embedding(block_size, dim_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(dim_embd)
        self.linear = nn.Linear(dim_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.linear(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) 
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            idx_crop = idx[:, -block_size:]
            logits, loss = self(idx_crop)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
#########################