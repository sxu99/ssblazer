# SSBlazer
SSBlazer is the first attempt to exploit the deep learning technique for SSB site prediction based on genome-wide single-strand break sequencing data with single-nucleotide resolution.

## Environment setup

### First, download the repository and create the environment.

```
git clone https://github.com/sxu99/ssblazer.git
cd ./ssblazer
conda env create -f environment.yml
```

### Then, activate the new environment.

```
conda activate ssblazer
```

## Prediction

### 1. Making predictions from loacl fasta file.

```
python prediction.py --file ./test.fa --batchsize 128
```

### 2. Use our online tool.

You can access [ssblazer server](https://proj.cse.cuhk.edu.hk/aihlab/ssblazer/) for the online version. 