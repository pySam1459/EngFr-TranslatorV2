import torch
from torch.amp import autocast
from numpy import load as np_load
from model import Translator, Config
from utils import load_encoding, DataLoader
from os.path import join, exists, isdir
from dotenv import dotenv_values
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LRScheduler:
    """Linear warmup and inverse square root decay."""
    def __init__(self, optimizer: torch.optim.Optimizer,
                 warmup_steps: int,
                 peak_lr: float,
                 current_step: int = 0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.current_step = current_step

    def step(self):
        self.current_step += 1
        lr = self.calculate_learning_rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def calculate_learning_rate(self):
        if self.current_step < self.warmup_steps:
            return self.peak_lr * self.current_step / self.warmup_steps
        else:
            return self.peak_lr / (self.current_step ** 0.5)


class LossGraph:
    EXT_LENGTH = 4096
    def __init__(self, history: torch.Tensor | None = None) -> None:
        if history is not None:
            self.loss = history
            self.i = history.shape[0]
        else:
            self.loss = torch.empty(LossGraph.EXT_LENGTH, dtype=torch.float16)
            self.i = 0
    
    def add(self, lossf: float) -> None:
        if self.i >= self.loss.shape[0]:
            self.loss = torch.cat((self.loss, torch.empty(LossGraph.EXT_LENGTH, dtype=torch.float16)))
            
        self.loss[self.i] = lossf
        self.i += 1

    def export(self) -> torch.Tensor:
        return self.loss[:self.i]


def main():
    encoding = load_encoding()
    config = Config(
        vocab_size = encoding.n_vocab,
        context_length = 512,
        d_model = 384,
        n_layer = 6,
        n_head = 6,
        dropout = 0.1)
    
    large_batch_size = 64#256
    small_batch_size = 16
    scale_factor = large_batch_size/small_batch_size
    grad_clip = None#1.0
    
    model = Translator(config)
    model = model.to(device)
    print(f"# of parameters: {model.n_params/1e6:.3f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = LRScheduler(optimizer, warmup_steps=1000, peak_lr=1e-4)
    loader = DataLoader(large_batch_size, config.context_length)
    
    resume_ckpt = "train1.pt"
    if resume_ckpt is None:
        loss_graph = LossGraph()
        print("Training from scratch.")
    else:
        resume_path = join(dotenv_values()["CKPT_PATH"], resume_ckpt)
        if isdir(resume_path): ## legacy
            checkpoint = torch.load(join(resume_path, f"ckpt_{resume_ckpt}.pt"))
            model.load_state_dict(checkpoint)
            loss_graph = LossGraph(np_load(join(resume_path, f"loss_{resume_ckpt}.npy")))
            print(f"Resumed from legacy {resume_ckpt}")
        
        elif exists(resume_path):
            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            loss_graph = LossGraph(checkpoint["loss"])
            lr_scheduler.current_step = checkpoint["lr_step"]
            loader.start_at(*checkpoint["loader_prog"])
            print(f"Resumed from {resume_ckpt}")
        else:
            print(f"Invalid checkpoint path {resume_path}")
            exit(1)

    ckpt_name = "train1"
    ckpt_path = dotenv_values()["CKPT_PATH"]
    ckpt_period = 512
    
    ctx = autocast(device_type="cuda", dtype=torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=True)   
    prog_bar = tqdm(loader, total=loader.total)
    for i, (xb, yb) in enumerate(prog_bar, start=1):
        for j in range(0, large_batch_size, small_batch_size):
            with ctx:
                _, loss = model(xb[j:j+small_batch_size], yb[j:j+small_batch_size])
                loss /= scale_factor
            scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        
        lossf = loss.item() * scale_factor
        prog_bar.set_description(f"{lossf:.7f}")
        loss_graph.add(lossf)
        if i % ckpt_period == 0: ## save checkpoint
            checkpoint = {}
            checkpoint["model"] = model.state_dict()
            checkpoint["optimizer"] = optimizer.state_dict()
            checkpoint["loss"] = loss_graph.export()
            checkpoint["lr_step"] = lr_scheduler.current_step
            checkpoint["loader_prog"] = loader.curr_prog()
            torch.save(checkpoint, join(ckpt_path, f"ckpt_{ckpt_name}.pt"))


if __name__ == "__main__":
    main()
