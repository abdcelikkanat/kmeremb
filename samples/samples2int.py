import pickle as pkl
import itertools


k = 4
filename = "3genomes_samples=1000"
input_file_path = f"./samples/{filename}.pkl"
output_file_path = f"./datasets/{filename}_intkmerseq_k={k}.pkl"

__LETTERS = ['A', 'C', 'G', 'T']
__LETTER2ID = {letter: idx for idx, letter in enumerate(__LETTERS)}
__KMERS = [''.join(kmer) for kmer in itertools.product(__LETTERS, repeat=k)]
__KMER2ID = {kmer: idx for idx, kmer in enumerate(__KMERS)}

with open(input_file_path, 'rb') as f:
    data = pkl.load(f)

sequences = []
for genome_idx, genome_name in enumerate(['geobacillus3', 'ecoli_k12_stat3', 'mruber1']):

    reads = data[genome_name]
    reads_num = len(reads)

    current_genome_seq = []
    for i in range(reads_num):

        current_read = reads[i]['read'][0]
        ascii = list(map(ord, reads[i]['ascii']))

        if len(current_read) <= k:
            continue

        current_read_seq = []
        for j in range(len(current_read) - k + 1):
            current_read_seq.append(__KMER2ID[current_read[j:j + k]])

        current_genome_seq.append(current_read_seq)

    sequences.append(current_genome_seq)

#
#
with open(output_file_path, 'wb') as f:
    pkl.dump(sequences, f)
#
# print(
#     samples[0],
#     samples[-1]
# )