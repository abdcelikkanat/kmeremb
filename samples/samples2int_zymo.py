import pickle as pkl
import itertools
import math

k = 4
filename = "zymohmw_samples=1000"
input_file_path = f"./samples/{filename}.pkl"
output_file_path = f"./datasets/{filename}_intkmerseq_k={k}_prob.pkl"

__LETTERS = ['A', 'C', 'G', 'T']
__LETTER2ID = {letter: idx for idx, letter in enumerate(__LETTERS)}
__KMERS = [''.join(kmer) for kmer in itertools.product(__LETTERS, repeat=k)]
__KMER2ID = {kmer: idx for idx, kmer in enumerate(__KMERS)}

with open(input_file_path, 'rb') as f:
    data = pkl.load(f)

print(
    len( data['zymohmw'][0] )
)
genome_name = 'zymohmw'
sequences = []
probs = []
for genome_idx in range(8):

    reads = data[genome_name][genome_idx]
    reads_num = len(reads)

    current_genome_seq = []
    current_genome_probs = []
    for i in range(reads_num):

        current_read = reads[i]['read'][0]
        ascii = list(map(ord, reads[i]['ascii']))

        if len(current_read) <= k:
            continue

        current_read_seq = []
        current_ascii_seq = []
        for j in range(len(current_read) - k + 1):
            current_read_seq.append(__KMER2ID[current_read[j:j + k]])
            current_ascii_seq.append( math.prod([1. - 10**(-(ascii[p]-33)/10) for p in range(j,j+k)]) )

        current_genome_seq.append(current_read_seq)
        current_genome_probs.append(current_ascii_seq)

    sequences.append(current_genome_seq)
    probs.append(current_genome_probs)


print(
    len(sequences)
)

# with open(output_file_path, 'wb') as f:
#     pkl.dump(sequences, f)

with open(output_file_path, 'wb') as f:
    pkl.dump({'sequences': sequences, 'probs':probs}, f)