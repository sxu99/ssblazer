import argparse
from ssblazer.my_model import SSBlazer
from Bio import SeqIO
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

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

def replace_char(s, index, new_char):
    s_list = list(s)
    s_list[index] = new_char
    s = "".join(s_list)
    return s

def get_seq(genome, pos, half_length):
    return str(genome[pos - half_length : pos + half_length + 1])

def get_seq_list(genome, pos, half_length, half_window_size):
    pos_s = list(range(pos - half_window_size, pos + half_window_size + 1))
    seq_list = []
    for p in pos_s:
        seq_list.append(get_seq(genome, p, half_length))
    return seq_list

def pred(model, seq_list):
    encoded_sequences = np.array([encode_single(seq) for seq in seq_list])
    batch = {"seq": torch.from_numpy(encoded_sequences).float()}
    break_probs = model.forward(batch)[1]
    return break_probs.detach().tolist()

def main():
    parser = argparse.ArgumentParser(description='Predict the effect of a genetic mutation.')
    parser.add_argument('--genome', type=str, help='Genome version.')
    parser.add_argument('--chr', type=str, help='Chromosome of interest.')
    parser.add_argument('--pos', type=int, help='Position of the mutation on the chromosome.')
    parser.add_argument('--ref', type=str, help='Reference allele.')
    parser.add_argument('--alt', type=str, help='Alternate allele.')
    args = parser.parse_args()

    model = SSBlazer(warmup=1, max_epochs=1, lr=1e-3)
    ckpt = torch.load("./ssblazer/ssblazer.pkl", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    chr = args.chr
    idx = args.pos - 1
    nt = args.ref
    snp = args.alt

    half_length = 125
    half_window_size = 125

    genome = SeqIO.read(f"./genome/{args.genome}/{chr}.fa", "fasta")
    genome_complement = genome.seq.complement()
    genome = str(genome.seq)
    genome_complement = str(genome_complement)

    if genome[idx].upper() == nt.upper():
        genome = genome
        print("Matched: +")
    elif genome_complement[idx].upper() == nt.upper():
        genome = genome_complement
        print("Matched: -")
    else:
        raise Exception(
            "The nucleotide at the specified index does not match the expected nucleotide in the reference genome. Please check your index and the original nucleotide."
        )
    mutant_genome = replace_char(s=genome, index=idx, new_char=snp)

    assert genome[idx].upper() == nt.upper()
    ref_seq_list = get_seq_list(genome, idx, half_length, half_window_size)
    mutant_seq_list = get_seq_list(
        mutant_genome, idx, half_length, half_window_size
    )

    with torch.no_grad():
        ref_seq_probs = pred(model, ref_seq_list)
        mutant_seq_probs = pred(model, mutant_seq_list)

        fig, axs = plt.subplots(2)

        axs[0].plot(ref_seq_probs)
        axs[0].set_title("Ref Seq Probs")
        axs[0].set_xticks([0, 125, 251])
        axs[0].set_xticklabels(["-125", "0", "+125"])
        axs[0].set_ylim([0, 1])

        axs[1].plot(mutant_seq_probs)
        axs[1].set_title("Mutant Seq Probs")
        axs[1].set_xticks([0, 125, 251])
        axs[1].set_xticklabels(["-125", "0", "+125"])
        axs[1].set_ylim([0, 1])
        plt.title("{}:{} {}->{}".format(chr, str(idx + 1), nt, snp))
        plt.subplots_adjust(hspace=0.5)
        os.makedirs("./snp", exist_ok=True)
        plt.savefig(
            "./snp/{}:{}_{}->{}.png".format(chr, str(idx + 1), nt, snp),
            dpi=600,
            bbox_inches="tight",
        )

        df = pd.DataFrame({
            'Ref Probs': ref_seq_probs,
            'Mutant Probs': mutant_seq_probs
        })

        df.to_csv('./snp/{}:{}_{}->{}.csv'.format(chr, str(idx + 1), nt, snp), index=False)


if __name__ == "__main__":
    main()