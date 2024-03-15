import pickle as pkl
import sys
import time
import itertools
import matplotlib.pyplot as plt
import numpy as np

normalize = True
dataset_sample_size = 1000
num_processes = 1
k = 4 # int(sys.argv[1]) #2
lr = 0.1
epoch_num = 1000
batch_size = 0
window_size = 32 #int(sys.argv[2]) #128  # 32
filename = f"zymohmw_samples={dataset_sample_size}_intkmerseq_k={k}"  # f"3genomes_samples=1000_intkmerseq_k={k}"
input_file_path = f"./datasets/{filename}.pkl"

init_time = time.time()
with open(input_file_path, 'rb') as f:
    sequences = pkl.load(f)
print(f"- Time to load the sequences: {time.time() - init_time}")

# Get the number of labels
class_num = len(sequences)
# Construct the labels
labels = [i for i in range(len(sequences)) for j in range(len(sequences[i]))]

# Convert the list of lists to a list
init_time = time.time()
sequences = list(itertools.chain.from_iterable(sequences))
print(f"- Time to convert the list of lists to a list: {time.time() - init_time}")

# Load the embeddings
with open(
        f"./embs/supervised_{filename}_{'' if normalize else 'un'}normalized_w={window_size}_"
        f"lr={lr}_epoch={epoch_num}_batch={batch_size}.pkl", 'rb'
) as f:
    embs = pkl.load(f)

# Get the embeddings of the sequences
# Convert sequences to embeddings
print(f"- Converting sequences to embeddings...")
init_time = time.time()
read_embs = np.zeros(shape=(len(sequences), embs.shape[1]), dtype=float)
for seq_idx, current_seq in enumerate(sequences):
    counts = np.bincount(current_seq, minlength=( 4** k ))
    read_embs[seq_idx] = np.dot(counts, embs)

    if normalize:
        read_embs[seq_idx] /= len(current_seq)
print(f"- Sequences converted to embeddings in {time.time() - init_time} seconds.")

# Plot the sequences embeddings
plt.figure(figsize=(10, 10))
for color_idx, color in enumerate(['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray']):
    plt.scatter(
        read_embs[color_idx * dataset_sample_size:(color_idx + 1) * dataset_sample_size, 0],
        read_embs[color_idx * dataset_sample_size:(color_idx + 1) * dataset_sample_size, 1], s=1, c=color
    )
plt.show()
