import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_encoding
from dataclasses import dataclass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.05)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.05)


@dataclass
class Config:
    vocab_size: int
    context_length: int
    d_model: int
    n_layer: int
    n_head: int
    dropout: float


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super(CausalSelfAttention, self).__init__()
        self.config = config
        
        self.attn = nn.Linear(config.d_model, 3*config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,T,C = x.size()
        out: torch.Tensor = self.attn(x)
        q, k, v = out.split(self.config.d_model, dim=2)
        q = q.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1,2)
        k = k.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1,2)
        v = v.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1,2)
        
        y: torch.Tensor = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.config.dropout, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.dropout(self.proj(y))
        return y


class FeedFoward(nn.Module):
    def __init__(self, config: Config) -> None:
        super(FeedFoward, self).__init__()
        self.config = config
        
        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return x


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.csa = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffd = FeedFoward(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.csa(self.ln1(x))
        x = x + self.ffd(self.ln2(x))
        return x


class Translator(nn.Module):
    def __init__(self, config: Config) -> None:
        super(Translator, self).__init__()
        
        self.tke = nn.Embedding(config.vocab_size, config.d_model)
        self.pse = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        self.ln1 = nn.LayerNorm(config.d_model)
        self.linear = nn.Linear(config.d_model, config.vocab_size)
        self.apply(init_) ## init weights

        self.config = config
        self.encoding = load_encoding()
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        B,T = x.size()
        tok_emb = self.tke(x)
        pse_emb = self.pse(torch.arange(T, device=device))
        x = tok_emb + pse_emb
        for block in self.blocks:
            x = block(x)
        
        logits: torch.Tensor = self.linear(self.ln1(x))
        if targets is None:
            return logits, None

        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), 
                               targets.view(B*T))
        return logits, loss

    def translate(self, idx: torch.Tensor,
                 max_len: int = 512,
                 temperature: float = 1.0) -> torch.Tensor:
        end = self.encoding.encode("<|endoftext|>", 
                                   allowed_special="all")[0]
        fr = []
        for _ in range(max_len):
            idx_cond = idx[:, -self.config.context_length:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            if idx_next.item() == end: break
            fr.append(idx_next)
        return torch.cat(fr, dim=1)
