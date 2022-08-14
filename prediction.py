import argparse
from gen_result import do_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SSBlazer parser.')
    parser.add_argument('--file', type=str,help='Path to the fasta file.',required=True)
    parser.add_argument('--batchsize', type=int,help='Batch size of the model.', default=128)
    args = parser.parse_args()

    do_pred(args.file,args.batchsize)
