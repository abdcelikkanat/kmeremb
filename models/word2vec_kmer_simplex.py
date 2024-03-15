import torch
import math
import time
import sys
import random
import pickle as pkl
import itertools
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Process, Pool, Array, Manager
from functools import partial

from sklearn.decomposition import PCA


class KMerEmb1D(torch.nn.Module):
    def __init__(self,  kmer_num, dim=2, latent_dim=7,
                 lr=0.1, epoch_num=100, batch_size = 1000,
                 device=torch.device("cpu"), verbose=False, seed=0):

        super(KMerEmb1D, self).__init__()

        self.__seed = seed
        self.__kmer_num = kmer_num
        self.__dim = dim
        self.__latent_dim = latent_dim
        self.__lr = lr
        self.__epoch_num = epoch_num
        self.__batch_size = batch_size
        self.__device = device
        self.__verbose = verbose

        self.__set_seed(seed)

        self.__A = torch.nn.Parameter(
            2 * torch.rand(size=(self.__latent_dim, self.__dim), device=self.__device) - 1, requires_grad=True
        )
        self.__embs = torch.nn.Parameter(
            2 * torch.rand(size=(self.__kmer_num, self.__latent_dim), device=self.__device) - 1, requires_grad=True
        )

        self.__optimizer = torch.optim.Adam(self.parameters(), lr=self.__lr)
        self.__loss = []
    def __set_seed(self, seed=None):

        if seed is not None:
            self._seed = seed

        random.seed(self._seed)
        torch.manual_seed(self._seed)

    def __compute_loss(self, x, degrees):

        A = torch.nn.functional.softmax(self.__A, dim=0)

        dist = torch.norm(
            torch.nn.functional.softmax(torch.index_select(self.__embs, 0, x[:, 0]),dim=1) @ torch.nn.functional.softmax(A, dim=0) - torch.nn.functional.softmax(torch.index_select(self.__embs, 0, x[:, 1]), dim=1) @ torch.nn.functional.softmax(A, dim=0),
            dim=1,
            p=1,
        )

        rate = torch.exp(-dist)

        return -(degrees * torch.log(rate) - rate).sum()

    def learn(self, x, degrees):

        for epoch in range(self.__epoch_num):

            # Shuffle data
            indices = torch.randperm(x.shape[0])
            x = x[indices]
            degrees = degrees[indices]

            epoch_loss = 0
            batch_size = self.__batch_size if self.__batch_size > 0 else x.shape[0]
            for i in range(0, x.shape[0], batch_size):

                batch_x = x[i:i+batch_size]
                batch_degrees = degrees[i:i+batch_size]

                if batch_x.shape[0] != batch_size:
                    continue

                self.__optimizer.zero_grad()

                batch_loss = self.__compute_loss(batch_x, batch_degrees)
                batch_loss.backward()
                self.__optimizer.step()

                self.__optimizer.zero_grad()

                epoch_loss += batch_loss.item()

            epoch_loss /= math.ceil(x.shape[0] / batch_size)

            if self.__verbose:
                print(f"epoch: {epoch}, loss: {epoch_loss}")

            self.__loss.append(epoch_loss)

    def get_emb(self):
        return torch.nn.functional.softmax(self.__embs, dim=1).detach().numpy(), torch.nn.functional.softmax(self.__A, dim=0).detach().numpy()

def compute_pair_counts(window_size, k, current_seq):

    pair_counts = torch.zeros(size=(4 ** k, 4 ** k), dtype=int)

    # for current_seq in sequences:
        # for current_seq in sequences:
    for i in range(len(current_seq)):

        # Get the center
        center = current_seq[i]

        for j in range(max(0, i - window_size), min(len(current_seq), i + window_size + 1)):
            if j == i:
                continue
            # Get the context
            context = current_seq[j]

            pair_counts[center, context] += 1

    return pair_counts

def kmerEmb2readEmb(embs, current_seq, normalize=True):

    dim = embs.shape[1]

    seq_emb = torch.zeros(size=(dim, ), dtype=torch.float)
    for i in range(len(current_seq)):
        seq_emb += embs[current_seq[i]]

    if normalize:
        seq_emb /= len(current_seq)

    return seq_emb



def update_array(shared_array, index, value):
    # Update the shared array at the specified index with the given value
    shared_array[index*5 + index] = value

    print(f"Index {index} updated to {value}")

if __name__ == "__main__":

    normalize = False
    dataset_sample_size = 1000
    num_processes = 1
    k = int(sys.argv[1]) #2
    lr = 0.01
    epoch_num = 1000
    batch_size = 0
    window_size = int(sys.argv[2]) #128  # 32
    filename = f"zymohmw_samples={dataset_sample_size}_intkmerseq_k={k}"  # f"3genomes_samples=1000_intkmerseq_k={k}"
    input_file_path = f"./datasets/{filename}.pkl"

    init_time = time.time()
    with open(input_file_path, 'rb') as f:
        sequences = pkl.load(f)
    print(f"- Time to load the sequences: {time.time() - init_time}")

    # Convert the list of lists to a list
    init_time = time.time()
    sequences = list(itertools.chain.from_iterable(sequences))
    print(f"- Time to convert the list of lists to a list: {time.time() - init_time}")

    # Get the total corpus size
    init_time = time.time()
    corpus_size = sum([len(seq) for seq in sequences])
    print(f"- Corpus size: {corpus_size} computed in {time.time()-init_time} seconds.")

    # Compute total center-context pairs by function
    init_time = time.time()
    pair_counts = np.zeros(shape=(4 ** k, 4 ** k), dtype=int)
    for current_seq in sequences:
        for i in range(len(current_seq)):
            # Get the center
            center = current_seq[i]
            for j in range(max(0, i - window_size), min(len(current_seq), i + window_size + 1)):
                if j == i:
                    continue
                context = current_seq[j]

                if center <= context:
                    pair_counts[center, context] += 1

    # Take add the upper triangular matrix to the lower triangular matrix
    pair_counts += np.triu(pair_counts, k=1).T
    print(f"- In total {pair_counts.sum() + np.triu(pair_counts, k=1).sum()} center-context pairs computed in {time.time() - init_time} seconds.")

    # Get the non-zero indices and values
    init_time = time.time()
    pair_counts = torch.from_numpy(pair_counts)
    x = torch.nonzero(pair_counts, as_tuple=False).T
    degrees = pair_counts[x[0], x[1]]
    # degrees -= degrees.min()
    print(x.shape, degrees.shape)
    print(f"- Non-zero indices and values computed in {time.time() - init_time} seconds.")
    #
    # Define the model
    kmer_model = KMerEmb1D(
        kmer_num=4 ** k, lr=lr, epoch_num=epoch_num, batch_size = 0, verbose=True
    )
    kmer_model.learn(x, degrees)
    # Get the embeddings
    embs, A = kmer_model.get_emb()

    # Define the output file name
    output_filename = f"simplex_{filename}_{'' if normalize else 'un'}normalized_w={window_size}_lr={lr}_epoch={epoch_num}_batch={batch_size}"

    # Save the embeddings
    with open(
            f"./new_embs/{output_filename}.pkl", 'wb'
    ) as f:
        pkl.dump(embs, f)

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

    # Apply pca to get 2d embeddings
    pca = PCA(n_components=2)
    read_embs = pca.fit_transform(read_embs)

    # Plot the sequences embeddings
    plt.figure(figsize=(10, 10))

    for color_idx, color in enumerate(['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray']):
        plt.scatter(
            read_embs[color_idx * dataset_sample_size:(color_idx + 1) * dataset_sample_size, 0],
            read_embs[color_idx * dataset_sample_size:(color_idx + 1) * dataset_sample_size, 1], s=1, c=color
        )

    plt.savefig(f"./new_figures/{output_filename}.png")



