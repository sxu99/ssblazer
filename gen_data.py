from ssblazer import dataloader,my_model
from torch.utils.data import DataLoader
from Bio import SeqIO

def get_dataloader(file_url,batch_size):
    fasta_sequences = SeqIO.parse(open(file_url),'fasta')
    seq=''
    for fasta in fasta_sequences:
        name, seq = fasta.id, str(fasta.seq)
        break

    dataset = dataloader.DatasetFromStr(seq)
    myloader = DataLoader(dataset, batch_size=batch_size,  collate_fn=dataloader.collate_fn,shuffle=False)
    
    return myloader 

