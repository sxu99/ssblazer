import argparse
from ssblazer.my_model import *
from ssblazer.dataloader import *
from Bio import SeqIO
from tqdm import tqdm
import torch
import numpy as np

nucleobase_mapping = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "a": [1, 0, 0, 0],
    "c": [0, 1, 0, 0],
    "g": [0, 0, 1, 0],
    "t": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
    "n": [0, 0, 0, 0],
}


def encode_single(seq):
    return [nucleobase_mapping.get(c, [0, 0, 0, 0]) for c in seq]


def sliding_window(sequence, window_size):
    for i in range(len(sequence) - window_size + 1):
        yield sequence[i : i + window_size]


def main(args):
    model = SSBlazer(warmup=1, max_epochs=1, lr=1e-3, length=251)

    BATCH_SIZE = args.batchsize
    ckpt = torch.load("./ssblazer/ssblazer.pkl", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    fa = args.infile
    genome = SeqIO.read(fa, "fasta")
    output_data = []
    with torch.no_grad():
        fragments = []
        for i, fragment in tqdm(enumerate(sliding_window(str(genome.seq), 251))):
            fragments.append(fragment)
            if len(fragments) == BATCH_SIZE:
                encoded_sequences = np.array([encode_single(seq) for seq in fragments])
                batch = {"seq": torch.from_numpy(encoded_sequences)}
                break_probs = model.forward(batch)[1]
                for j, break_prob in enumerate(break_probs):
                    output_data.append(
                        f"chr1\t{i - BATCH_SIZE + j + 126}\t{i - BATCH_SIZE + j + 126 + 1}\t{break_prob.item()}\n"
                    )
                fragments = []
        # Handle remaining sequences if they are less than BATCH_SIZE
        if fragments:
            encoded_sequences = np.array([encode_single(seq) for seq in fragments])
            batch = {"seq": torch.from_numpy(encoded_sequences)}

            break_probs = model.forward(batch)[1]

            for j, break_prob in enumerate(break_probs):
                output_data.append(
                    f"chr1\t{i - len(fragments) + j + 126}\t{i - len(fragments) + j + 126 + 1}\t{break_prob.item()}\n"
                )

    with open(args.outfile, "w") as f:
        f.writelines(output_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, help="Input FASTA file")
    parser.add_argument("--outfile", type=str, help="Output BED file")
    parser.add_argument("--batchsize", type=int, help="Batch size for processing")
    args = parser.parse_args()
    main(args)
