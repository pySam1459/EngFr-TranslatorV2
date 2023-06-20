import torch
from tiktoken import Encoding, get_encoding
from os import listdir
from os.path import join
from dotenv import dotenv_values
from typing import Iterator


__all__ = ["load_encoding", "DataLoader"]


def load_encoding() -> Encoding:
    _gpt2_encoding = get_encoding("gpt2")
    _gpt2_n_vocab = _gpt2_encoding.n_vocab
    return Encoding(
        name="gpt2_en_fr",
        pat_str=_gpt2_encoding._pat_str,
        mergeable_ranks=_gpt2_encoding._mergeable_ranks,
        special_tokens={
            **_gpt2_encoding._special_tokens,
            "<|start|>": _gpt2_n_vocab,
            "<|equals|>": _gpt2_n_vocab+1,
        },
        explicit_n_vocab=_gpt2_n_vocab+2
    )


class DataLoader:
    def __init__(self, batch_size: int, context_length: int) -> None:
        self.batch_size = batch_size
        self.context_length = context_length
        
        dataset_path = dotenv_values(".env")["DATASET_PATH"]
        self.partition_path = join(dataset_path, "partitionedv2")
        self.partitions = listdir(self.partition_path)
        self.batch_per_part = 1/batch_size
        self.total = self.get_total()
        self._file_idx, self._part_idx = 0, 0
       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_total(self) -> int:
        total = 0
        for file in self.partitions:
            partition = torch.load(join(self.partition_path, file))
            total += int(partition.shape[0] * self.batch_per_part)
        return total

    def start_at(self, file_idx: int, part_idx: int) -> None:
        self._file_idx = file_idx
        self._part_idx = part_idx
    
    def curr_prog(self) -> tuple[int, int]:
        return self._file_idx, self._part_idx
    
    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for i, file in enumerate(self.partitions[self._file_idx:]):
            self._file_idx = i
            partition = torch.load(join(self.partition_path, file))
            part_length = partition.shape[0]
            for k in range(self._part_idx, int(part_length * self.batch_per_part)):
                self._part_idx = k
                ix = torch.randint(0, part_length-self.context_length, (self.batch_size,))
                xb = [partition[i:i+self.context_length] for i in ix]
                yb = [partition[i+1:i+self.context_length+1] for i in ix]
                xb = torch.stack(xb).to(self.device)
                yb = torch.stack(yb).to(self.device)
                yield (xb, yb)
        yield None