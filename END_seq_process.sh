#!/bin/bash

# Define paths to input files and outputs (replace placeholders)
INPUT_READS="<path_to_input_reads>"
OUTPUT_PREFIX="<output_prefix>"
BOWTIE_INDEX_HG19="<bowtie_index_hg19>"
BOWTIE_INDEX_MM10="<bowtie_index_mm10>"
BOWTIE_INDEX_RN6="<bowtie_index_rn6>"
CHROM_SIZES="<path_to_chrom_sizes>"

# Define Bowtie parameters for END-seq
BOWTIE_PARAMS="-n 3 -l 50 -k 1"

# Bowtie alignment for END-seq
# Raw data ref GSM5100382 and 
# Assuming the reads are from human i3Neuron or iMuscle
bowtie $BOWTIE_INDEX_HG19 $BOWTIE_PARAMS -q $INPUT_READS | \
samtools view -Sb - | \
samtools sort - -o ${OUTPUT_PREFIX}_END_seq_sorted.bam

# Convert sorted BAM to BED format
bedtools bamtobed -i ${OUTPUT_PREFIX}_END_seq_sorted.bam > ${OUTPUT_PREFIX}_END_seq.bed

# Generate bigwig files for positive and negative strands
for strand in pos neg; do
    genomeCoverageBed -bg -strand $strand -i ${OUTPUT_PREFIX}_END_seq.bed -g $CHROM_SIZES > ${OUTPUT_PREFIX}_END_seq_${strand}.bedGraph
    bedGraphToBigWig ${OUTPUT_PREFIX}_END_seq_${strand}.bedGraph $CHROM_SIZES ${OUTPUT_PREFIX}_END_seq_${strand}.bw
done

echo "END-seq processing complete."
