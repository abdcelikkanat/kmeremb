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


######
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
read_list = random.sample(sequences, reads_num)
######

kmer_list = []
for read_idx in range(reads_num):
    read = read_list[read_idx]
    read_kmer_profile = read2kmerseq(read, k)

    kmer_list.append(read_kmer_profile)


edit_dist_list, l1_dist_list = [], []
for i in range(reads_num):
    for j in range(i+1, reads_num):

        edit_dist_list.append( get_edit_distance(read_list[i], read_list[j]) )
        l1_dist_list.append( np.linalg.norm(kmer_list[i] - kmer_list[j], ord=1) )

edit_dist_list, l1_dist_list = np.array(edit_dist_list), np.array(l1_dist_list)

plt.figure()
plt.plot(
    edit_dist_list, l1_dist_list, '.', color='darkblue', markersize=2)
plt.xlabel('Edit distance')
plt.ylabel('L1 distance')
plt.savefig(f"./new_figures/edit_vs_l1_distances_real_data.png")
