# SSBlazer

SSBlazer is a pioneering tool that utilizes deep learning to predict Single Strand Break (SSB) sites based on genome-wide single-strand break sequencing data with single-nucleotide resolution. 

Access the complete [SSBlazer Manual](https://sxu99.gitbook.io/ssblazer/) for more details.

## Environment Setup

Follow these steps to set up the SSBlazer environment:

1. Clone the repository and navigate into the directory:

    ```
    git clone https://github.com/sxu99/ssblazer.git
    cd ./ssblazer
    ```

2. Create a new environment using the provided environment file:

    ```
    conda env create -f environment.yml
    ```

3. Activate the newly created environment:

    ```
    conda activate ssblazer
    ```

## Use Cases

### Genome Prediction

To annotate Single-Strand Break (SSB) sites on a genome, use the `genome_pred.py` script. This script reads a FASTA file, predicts SSB sites, and outputs the predictions in BED format:

```
python genome_pred.py --in chr1.fa --out chr1_annotated.bed --batchsize 128
```

- `--in chr1.fa`: Specifies the input FASTA file for the genome to be annotated.
- `--out chr1_annotated.bed`: Specifies the output BED file where the annotations will be written.
- `--batchsize 128`: Sets the batch size to 128 for processing. Adjust this value depending on your machine's resources.

### Mutation Analysis

Perform mutation prediction using the following command:

```
python pred_mutation.py --genome hg19 --chr chr14 --pos 73659501 --ref 'T' --alt 'C'
```

- `--genome hg19`: Specifies the genome version.
- `--chr chr14`: Represents the chromosome of interest.
- `--pos 73659501`: Identifies the position of the mutation on the chosen chromosome.
- `--ref 'T'`: Indicates the reference allele, i.e., the original nucleotide at the mutation site.
- `--alt 'C'`: Designates the alternate allele, which is the mutated nucleotide.

### Train a New Model

Train a new model using your datasets:

```
python train_from_scratch.py --train train.csv --test test.csv
```

The model weights will be saved in the `./models` directory. After the model is trained, you can use it to predict break sites on new data by loading the model weights.

## Web Server

For easier access and usage, check out the [SSBlazer Web Server](https://proj.cse.cuhk.edu.hk/aihlab/ssblazer/).
