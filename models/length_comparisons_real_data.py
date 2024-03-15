import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import time

# Parameters
seed = 10
read_length = 10000
k = 4
iter_num = 10000
reads_num = 100

# Set seed
random.seed(seed)
np.random.seed(seed)

# Global variables
__letters = ['A', 'C', 'G', 'T']
kmer2Id = {kmer: idx for idx, kmer in enumerate([''.join(kmer) for kmer in itertools.product(__letters, repeat=k)])}

# Convert read to kmer profile
def read2kmerseq(read, k):

    profile = np.zeros(4**k, dtype=int)
    for i in range(len(read)-k+1):
        profile[kmer2Id[read[i:i+k]]] += 1

    return profile

def get_edit_distance(read1, read2):
    assert len(read1) == len(read2), f"Reads have different lengths: {len(read1)} and {len(read2)}"

    return sum([read1[i] != read2[i] for i in range(len(read1))])


with open('./samples/zymohmw_samples=1000.pkl', 'rb') as f:
    data = pkl.load(f)


sequences = []
for genome_idx in range(8):

    reads = data['zymohmw'][genome_idx]
    reads_num = len(reads)

    current_genome_seq = []
    current_genome_probs = []
    for i in range(reads_num):

        current_read = reads[i]['read'][0]

        if len(current_read) >= read_length:
            sequences.append(current_read[:read_length])

# Sample reads_num reads
sequences = random.sample(sequences, reads_num)

edit_dist_list, l1_dist_list = [], []
for read_idx in range(reads_num):

    current_edit_dist_list, current_l1_dist_list = [], []
    # Generate a random read
    read = sequences[read_idx]
    read_kmer_profile = read2kmerseq(read, k)

    read_mut = list(str(read))
    for iter in range(iter_num):

        # Add a random mutation
        pos = random.randint(0, read_length-1)
        read_mut[pos] = __letters[random.randint(0, 3)]

        # Convert the mutated read to a kmer profile
        read_mut_kmer_profile = read2kmerseq(''.join(read_mut), k)

        current_edit_dist_list.append( get_edit_distance(read, read_mut) )
        current_l1_dist_list.append( np.linalg.norm(read_kmer_profile - read_mut_kmer_profile, ord=1) )
    # Convert the list read_mut to string
    read_mut = ''.join(read_mut)

    edit_dist_list.append(current_edit_dist_list)
    l1_dist_list.append(current_l1_dist_list)

plt.figure()
plt.title(f"Edit vs L1 distances for {reads_num} reads of length {read_length}.")
edit_dist_list_mean, edit_dist_list_std = np.mean(edit_dist_list, axis=0), np.std(edit_dist_list, axis=0)
l1_dist_list_mean, l1_dist_list_std = np.mean(l1_dist_list, axis=0), np.std(l1_dist_list, axis=0)
plt.plot(range(iter_num), edit_dist_list_mean, '-', color='darkblue', label='Edit distance')
plt.fill_between(range(iter_num), edit_dist_list_mean-edit_dist_list_std, edit_dist_list_mean+edit_dist_list_std, color='darkblue', alpha=0.3)
plt.plot(range(iter_num), l1_dist_list_mean, '-', color='darkred', label='L1 distance')
plt.fill_between(range(iter_num), l1_dist_list_mean-l1_dist_list_std, l1_dist_list_mean+l1_dist_list_std, color='darkred', alpha=0.3)
plt.xlabel('Number of mutation operations')
plt.ylabel('Distance')
plt.legend()
# plt.show()
plt.savefig(f"./new_figures/edit_vs_l1_distances_{reads_num}_reads_{read_length}_length_iter_{iter_num}_real_dataset.png")
