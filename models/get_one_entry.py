import pickle as pkl
import itertools
import math
import time
import matplotlib.pyplot as plt

filename = "zymohmw_samples=1000"
input_file_path = f"./samples/{filename}.pkl"

__LETTERS = ['A', 'C', 'G', 'T']
__LETTER2ID = {letter: idx for idx, letter in enumerate(__LETTERS)}

with open(input_file_path, 'rb') as f:
    data = pkl.load(f)

print(
    len( data['zymohmw'][0] )
)
genome_name = 'zymohmw'
sequences = []
ascii_seqs = []
probs = []
for genome_idx in range(8):

    reads = data[genome_name][genome_idx]
    reads_num = len(reads)

    current_genome_seq = []
    current_genome_ascii = []
    current_genome_probs = []
    for i in range(reads_num):

        current_read = reads[i]['read'][0]
        ascii = list(map(ord, reads[i]['ascii']))

        # Convert the current_read to ascii list
        current_prob = [1. - 10**(-(ascii[l]-33)/10) for l in range(len(ascii))]

        current_genome_seq.append(current_read)
        current_genome_ascii.append(ascii)
        current_genome_probs.append(current_prob)

    sequences.append(current_genome_seq)
    probs.append(current_genome_probs)
    ascii_seqs.append(current_genome_ascii)

# Convert the list of lists to a list
init_time = time.time()
sequences = list(itertools.chain.from_iterable(sequences))
probs = list(itertools.chain.from_iterable(probs))
ascii_seqs = list(itertools.chain.from_iterable(ascii_seqs))

print(f"- Time to convert the list of lists to a list: {time.time() - init_time}")


print(
    len(sequences)
)

print(len(sequences[0]))

i = 100
with open(f"./one_entry_{i}.csv", 'w') as f:
    # for i in range(len(sequences)):

    f.write(f"{sequences[i]}\n")
    f.write(f"{','.join(map(str, ascii_seqs[i]))}\n")
    f.write(f"{','.join(map(str, probs[i]))}\n")

# plt.figure()
# # Plot the first 1000 probabilities
# plt.plot(probs[0], '.', markersize=1)
# plt.show()