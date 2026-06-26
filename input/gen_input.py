import torch
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Mamba models on multiple GPUs")
    parser.add_argument("-batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("-seq_len", type=int, default=1024, help="Sequence length")
    args = parser.parse_args()

    input_ids = torch.randint(0, 50280, (args.batch_size, args.seq_len)).to("cuda:0")

    torch.save(input_ids, f"input/input_ids.pt")