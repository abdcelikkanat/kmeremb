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
from sklearn.utils import shuffle



class Word2VecKMerEmb(torch.nn.Module):
    def __init__(self,  kmer_num, class_num, dim=2,
                 lr=0.1, epoch_num=100, batch_size = 1000,
                 device=torch.device("cpu"), verbose=False, seed=0):

        super(Word2VecKMerEmb, self).__init__()

        self.__seed = seed
        self.__kmer_num = kmer_num
        self.__class_num = class_num
        self.__dim = dim
        self.__lr = lr
        self.__epoch_num = epoch_num
        self.__batch_size = batch_size
        self.__device = device
        self.__verbose = verbose

        self.__set_seed(seed)

        self.__embs = torch.nn.Parameter(
            2 * torch.rand(size=(self.__kmer_num, self.__dim), device=self.__device) - 1, requires_grad=True
        )

        self.__optimizer = torch.optim.Adam(self.parameters(), lr=self.__lr)
        self.__loss = []

        self.__softmax_weights = torch.nn.Parameter(
            2 * torch.rand(size=(self.__class_num, self.__dim), device=self.__device) - 1, requires_grad=True
        )
    def __set_seed(self, seed=None):

        if seed is not None:
            self._seed = seed

        random.seed(self._seed)
        torch.manual_seed(self._seed)

    def __compute_loss(self, reads, read_labels):

        loss = 0
        for current_read, current_label in zip(reads, read_labels):

            bincounts = torch.bincount(torch.asarray(current_read), minlength=self.__kmer_num).to(torch.float)
            read_emb = bincounts @ self.__embs

            output = torch.log_softmax( self.__softmax_weights @ read_emb,dim=0)
            # output = torch.log_softmax(- torch.sum((self.__softmax_weights - read_emb.unsqueeze(0))**2, dim=1), dim=0)


            # Compute the loss
            loss += -output[current_label]

            # loss += -torch.sum((read_emb - self.__softmax_weights[current_label])**2) / self.__dim

        return loss

    def learn(self, reads, read_labels):

        for epoch in range(self.__epoch_num):

            reads, read_labels = shuffle(reads, read_labels)

            epoch_loss = 0
            batch_size = self.__batch_size if self.__batch_size > 0 else len(reads)
            for i in range(0, len(reads), batch_size):

                batch_reads = reads[i:i+batch_size]
                batch_labels = read_labels[i:i+batch_size]

                if len(batch_reads) != batch_size:
                    continue

                self.__optimizer.zero_grad()

                batch_loss = self.__compute_loss(batch_reads, batch_labels)
                batch_loss.backward()
                self.__optimizer.step()

                self.__optimizer.zero_grad()

                epoch_loss += batch_loss.item()

            epoch_loss /= math.ceil(len(reads) / batch_size)

            if self.__verbose:
                print(f"epoch: {epoch}, loss: {epoch_loss}")

            self.__loss.append(epoch_loss)

    def get_emb(self):
        return self.__embs.detach().numpy()

    def get_sofmax_weights(self):
        return self.__softmax_weights.detach().numpy()


def update_array(shared_array, index, value):
    # Update the shared array at the specified index with the given value
    shared_array[index*5 + index] = value

    print(f"Index {index} updated to {value}")

if __name__ == "__main__":

    normalize = True
    dataset_sample_size = 1000
    num_processes = 1
    k = int(sys.argv[1]) #4
    lr = 0.1
    epoch_num = 1000
    batch_size = 0
    window_size = int(sys.argv[2]) #32 #128  # 32
    filename = f"zymohmw_samples={dataset_sample_size}_intkmerseq_k={k}"  # f"3genomes_samples=1000_intkmerseq_k={k}"
    input_file_path = f"./datasets/{filename}.pkl"

    init_time = time.time()
    with open(input_file_path, 'rb') as f:
        sequences = pkl.load(f)
    print(f"- Time to load the sequences: {time.time() - init_time}")

    class_num = len(sequences)
    labels = [i for i in range(class_num) for j in range(len(sequences[i]))]
    print(f"- Number of classes: {class_num}")

    # Convert the list of lists to a list
    init_time = time.time()
    sequences = list(itertools.chain.from_iterable(sequences))
    print(f"- Time to convert the list of lists to a list: {time.time() - init_time}")

    # Get the total corpus size
    init_time = time.time()
    corpus_size = sum([len(seq) for seq in sequences])
    print(f"- Corpus size: {corpus_size} computed in {time.time()-init_time} seconds.")

    # Define the model
    w2v_kmer_model = Word2VecKMerEmb(
        kmer_num=4 ** k, class_num=class_num, dim=2, lr=lr, epoch_num=epoch_num, batch_size = 0, verbose=True
    )
    w2v_kmer_model.learn(reads=sequences, read_labels=labels)
    # Get the embeddings
    embs = w2v_kmer_model.get_emb()
    softmax_weights = w2v_kmer_model.get_sofmax_weights()

    # Save the embeddings
    with open(
            f"./embs/supervised_{filename}_{'' if normalize else 'un'}normalized_w={window_size}_"
            f"lr={lr}_epoch={epoch_num}_batch={batch_size}.pkl", 'wb'
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

    # Plot the sequences embeddings
    plt.figure(figsize=(10, 10))

    for color_idx, color in enumerate(['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray']):
        plt.scatter(
            read_embs[color_idx * dataset_sample_size:(color_idx + 1) * dataset_sample_size, 0],
            read_embs[color_idx * dataset_sample_size:(color_idx + 1) * dataset_sample_size, 1], s=1, c=color
        )
        plt.scatter(softmax_weights[color_idx, 0], softmax_weights[color_idx, 1], s=1, c=color, marker='x')

    # plt.show()
    plt.savefig(
        f"./figures/supervised_{filename}_{'' if normalize else 'un'}normalized_w={window_size}_"
        f"lr={lr}_epoch={epoch_num}_batch={batch_size}.png"
    )
    '''
    '''

