import torch
import h5py
from utils import load_encoding
from os.path import join
from dotenv import dotenv_values
from tqdm import tqdm
from multiprocessing import Process, Queue


dataset_path = dotenv_values(".env")["DATASET_PATH"]


def load_worker(chunksize: int, out_queue: Queue) -> None:
    filename = "en-fr_tokenized.h5"
    with h5py.File(join(dataset_path, filename), "r") as f_in:
        total = f_in["en"].shape[0]
        out_queue.put(total)
        for k in range(0, total//chunksize):
            chunksize_ = min(chunksize, total-k)
            en_chunk = f_in["en"][k*chunksize:k*chunksize+chunksize_]
            fr_chunk = f_in["fr"][k*chunksize:k*chunksize+chunksize_]
            out_queue.put((k, en_chunk, fr_chunk))

    out_queue.put(None)


def main():
    encoding = load_encoding()
    start, equals, stop = torch.tensor(encoding.encode("<|start|><|equals|><|endoftext|>",
                               allowed_special="all")).split(1)
    
    scale_factor = 10
    partition_size = 1000000
    chunksize = partition_size // scale_factor
    dataset = []
    out_queue = Queue()
    process = Process(target=load_worker, args=(chunksize, out_queue), daemon=True)
    process.start()

    total = out_queue.get()
    prog_bar = tqdm(total=total)
    while (data := out_queue.get()) is not None:
        k, en_chunk, fr_chunk = data
        for i, (en, fr) in enumerate(zip(en_chunk, fr_chunk)):
            prog_bar.update(1)
            if (i+1) % 1000 == 0:
                prog_bar.refresh()

            en_length, fr_length = en.shape[0], fr.shape[0]
            if not (en_length * 0.5 < fr_length < en_length * 2): continue
            
            parts = [start, torch.from_numpy(en), equals, torch.from_numpy(fr), stop]
            dataset.extend(parts)            

        if k % scale_factor == 0: 
            torch.save(torch.cat(dataset), join(dataset_path, "partitionedv2", f"{k+scale_factor}.pt"))
            dataset.clear()

    process.join()


if __name__ == "__main__":
    main()
