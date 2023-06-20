import torch
from model import Translator, Config
from utils import load_encoding
from dotenv import dotenv_values
from os.path import join


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    encoding = load_encoding()
    config = Config(
        vocab_size = encoding.n_vocab,
        context_length = 512,
        d_model = 384,
        n_layer = 6,
        n_head = 6,
        dropout = 0.1)
    equals_tok = encoding.encode("<|equals|>", allowed_special="all")[0]

    model = Translator(config)
    model = model.to(device)
    
    ckpt_name = "train1"
    ckpt_path = join(dotenv_values()["CKPT_PATH"], f"ckpt_{ckpt_name}.pt")
    checkpoint = torch.load(ckpt_path)

    model.load_state_dict(checkpoint["model"])
    model.eval()
    print("State Dict loaded.")
    
    while True:
        in_text = input(">> ")
        in_tok = encoding.encode(in_text)
        in_tok.append(equals_tok)
        in_tok = torch.tensor(in_tok, dtype=torch.long, device=device).view(1, -1)
        out_tok = model.translate(in_tok)
        out_text = encoding.decode(out_tok[0].tolist())
        print(out_text)


if __name__ == "__main__":
    main()
