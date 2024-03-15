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



class Word2VecKMerEmb(torch.nn.Module):
    def __init__(self,  kmer_num, dim=2,
                 lr=0.1, epoch_num=100, batch_size = 1000,
                 device=torch.device("cpu"), verbose=False, seed=0):

        super(Word2VecKMerEmb, self).__init__()

        self.__seed = seed
        self.__kmer_num = kmer_num
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
    def __set_seed(self, seed=None):

        if seed is not None:
            self._seed = seed

        random.seed(self._seed)
        torch.manual_seed(self._seed)

    def __compute_loss(self, x, degrees):

        dist = torch.norm(
            torch.index_select(self.__embs, 0, x[:, 0]) - torch.index_select(self.__embs, 0, x[:, 1]),
            dim=1
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
        return self.__embs.detach().numpy()



# x = torch.tensor([
#     (0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1,4), (2, 3), (2, 4), (3, 4)
# ], dtype=torch.long)
# degrees = torch.tensor([10, 10, 10, 10, 5, 5, 5, 1, 1, 1], dtype=torch.float)

'''
k = 4
window_size = 8 #32
filename = f"zymohmw_samples=1000_intkmerseq_k={k}"  #f"3genomes_samples=1000_intkmerseq_k={k}"
input_file_path = f"./datasets/{filename}.pkl"

with open(input_file_path, 'rb') as f:
    sequences = pkl.load(f)

# Convert the list of lists to a list
sequences = list(itertools.chain.from_iterable(sequences))


# Construct the counts
counts = np.zeros(shape=(4**k, 4**k), dtype=int)
for current_seq in sequences:
    for i in range(len(current_seq)):

        # Get the center
        center = current_seq[i]

        for j in range(max(0, i - window_size), min(len(current_seq), i + window_size + 1)):
            if j == i:
                continue
            # Get the context
            context = current_seq[j]
            # Increment the count
            counts[center, context] += 1

            # if (center, context) not in corpus:
            #     corpus[(center, context)] = 1
            # else:
            #     corpus[(center, context)] += 1

print(counts.min(), counts.max())
'''

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

    dataset_sample_size = 100
    num_processes = 1
    k = 8
    lr = 0.01
    epoch_num = 1000
    batch_size = 0
    window_size = 8  # 32
    filename = f"zymohmw_samples={dataset_sample_size}_intkmerseq_k={k}"  # f"3genomes_samples=1000_intkmerseq_k={k}"
    input_file_path = f"./datasets/{filename}.pkl"

    with open(input_file_path, 'rb') as f:
        sequences = pkl.load(f)

    # Convert the list of lists to a list
    sequences = list(itertools.chain.from_iterable(sequences))

    # Compute the counts in parallel
    pair_counts = torch.zeros(size=(4 ** k, 4 ** k), dtype=int)
    with Pool(processes=num_processes) as pool:
        results = pool.map(partial(compute_pair_counts, window_size, k), sequences)

        for result in results:
            pair_counts += result

    print(f"Pair counts computed.")

    # Get the non-zero indices and values
    x = torch.nonzero(pair_counts, as_tuple=False).T
    degrees = pair_counts[x[0], x[1]]

    print(x.shape, degrees.shape)

    # Define the model
    w2v_kmer_model = Word2VecKMerEmb(
        kmer_num=4 ** k, lr=lr, epoch_num=epoch_num, batch_size = 0, verbose=True
    )
    w2v_kmer_model.learn(x, degrees)
    # Get the embeddings
    embs = w2v_kmer_model.get_emb()

    # # Save the embeddings
    # with open(f"./{filename}_embs.pkl", 'wb') as f:
    #     pkl.dump(embs, f)
    # with open(f"./{filename}_embs.pkl", 'rb') as f:
    #     embs = pkl.load(f)

    # Convert sequences to embeddings
    with Pool(processes=num_processes) as pool:
        results = pool.map(partial(kmerEmb2readEmb, embs, normalize=True), sequences)

        embs_sequences = torch.stack(results)

    print(embs_sequences.shape)

    # Plot the sequences embeddings
    plt.figure(figsize=(10, 10))

    for color_idx, color in enumerate(['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray']):
        plt.scatter(
            embs_sequences[color_idx * dataset_sample_size:(color_idx + 1) * dataset_sample_size, 0],
            embs_sequences[color_idx * dataset_sample_size:(color_idx + 1) * dataset_sample_size, 1], s=1, c=color
        )

    # plt.show()
    plt.savefig(f"./figures/1___{filename}_w={window_size}_lr={lr}_epoch={epoch_num}_batch={batch_size}.png")


    '''
    array_size = 5

    # Create a synchronized NumPy array using the manager
    # shared_array = Array('d', array_size)
    #
    # # Create a NumPy array from the shared array
    # np_array = np.frombuffer(shared_array.get_obj(), dtype=int)
    # Set the array to zero
    # np_array[:] = 0
    # Set the shared array to zero
    # shared_array[:] = 0
    np_array = np.zeros(shape=(array_size, array_size), dtype=float)
    shared_array = Array('d', np_array.ravel())

    # Reshape the shared array to match the NumPy array
    # shared_array = np.frombuffer(shared_array.get_obj(), dtype=float).reshape(np_array.shape)

    # # get ctype for our array
    # ctype = as_ctypes(np_array)
    # # create ctype array initialized from our array
    # shared_array = Array(ctype._type_, np_array, lock=False)

    print("Initial array:", np_array)
    print("Initial shared array:", shared_array)
    # Run the update function in parallel
    processes = []
    for i in range(array_size):
        p = Process(target=update_array, args=(shared_array, i, i * 2))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Final array:", np_array)
    # print("Final shared array:", shared_array[:array_size, :array_size])
    # Convert the shared array to a int NumPy array
    # np_array = np.frombuffer(shared_array.get_obj(), dtype=int)
    # print("Final array:", np_array)
    # The last two lines gives the following error:
    # ValueError: buffer size must be a multiple of element size
    # The following code is a workaround for this error
    np_array = np.frombuffer(shared_array.get_obj(), dtype=float).reshape(np_array.shape)
    print("Final array:", np_array)
    '''

    # # Create a pool of processes
    # pool = Pool(processes=num_processes)
    #
    # # Use the pool to asynchronously update the shared array in parallel
    # # Pass each task as a tuple
    # tasks = [(shared_array, i, i * 2) for i in range(array_size)]
    # pool.starmap(update_array, tasks)
    #
    # # Close the pool and wait for the worker processes to finish
    # pool.close()
    # pool.join()







    # Print the updated shared array
    # print("Updated shared array:", np_array)

    '''
    # Construct the counts
    # unshared_pair_counts = np.zeros(shape=(4 ** k, 4 ** k), dtype=int)
    # pair_counts = Array('d', unshared_pair_counts)
    # Convert unshared_pair_counts to shared memory
    shared_pair_counts = Array('d', (4 ** k) * (4 ** k))
    pair_counts = np.frombuffer(shared_pair_counts.get_obj(), dtype=int).reshape(4 ** k, 4 ** k)

    # # Compute the pair_counts in parallel
    # with Pool(8) as p:
    #     p.map(partial(compute_pair_counts, pair_counts, window_size), sequences)
    pool = Pool(processes=4)
    tasks = [(pair_counts, window_size, sequences[i: i+2]) for i in range(4)]
    pool.starmap(compute_pair_counts, tasks)
    pool.close()
    pool.join()
    print(pair_counts[0, 0])
    '''

    # The code above gives the following error:
    # RuntimeError: SynchronizedArray objects should only be shared between processes through inheritance
    # The following code is a workaround for this error
    # pair_counts = np.frombuffer(pair_counts.get_obj()).reshape(unshared_pair_counts.shape)

    # Compute the pair_counts in parallel
    # processes = []
    # for i in range(8):
    #     p = Process(
    #         target=compute_pair_counts,
    #         args=(pair_counts, window_size, sequences[math.floor(i*len(sequences)/8):math.floor((i+1)*len(sequences)/8)],)
    #     )
    #     p.start()
    #     processes.append(p)
    #
    # for p in processes:
    #     p.join()


    # p = Process(target=compute_pair_counts, args=(pair_counts, window_size, sequences[0]))
    # p = Process(
    #     target=partial(compute_pair_counts, pair_counts, window_size),
    #     args=(sequences,)
    # )
    # p.start()
    # p.join()


    # start_time = time.time()
    # with Pool(8) as p:
    #     p.map(partial(compute_pair_counts, pair_counts, window_size), sequences)
    # print(
    #     f"--- {time.time() - start_time} seconds ---"
    # )
    #
    # print(pair_counts.min(), pair_counts.max())


# print(
#     f"There are {len(corpus)} unique center context pairs out of {4**k} possible pairs.",
# )

# #
# # Convert the dictionary to a list of keys and a list of the corresponding values
# x = torch.tensor(list(corpus.keys()), dtype=torch.long)
# degrees = torch.tensor(list(corpus.values()), dtype=torch.float)
#
#
# # Define the model
# w2v_kmer_model = Word2VecKMerEmb(
#     kmer_num=4**k, lr=0.1, epoch_num=100, batch_size = len(x), verbose=True
# )
# w2v_kmer_model.learn(x, degrees)
# embs = w2v_kmer_model.get_emb()
#
# print(
#     embs.shape
# )

'''
# Convert the list of lists to a list
combined_sequences = [item for sublist in sequences for item in sublist]

__LETTERS = ['A', 'C', 'G', 'T']
__LETTER2ID = {letter: idx for idx, letter in enumerate(__LETTERS)}
__KMERS = [''.join(kmer) for kmer in itertools.product(__LETTERS, repeat=k)]
__KMER2ID = {kmer: idx for idx, kmer in enumerate(__KMERS)}


corpus = {}
for seq in combined_sequences:
    for i in range(len(seq)):

        center = seq[i]

        for j in range(i - window_size, i + window_size + 1):
            if j < 0 or j >= len(seq) or j == i:
                continue

            context = seq[j]

            if (center, context) not in corpus:
                corpus[(center, context)] = 1
            else:
                corpus[(center, context)] += 1

print(
    len(corpus), len(__KMERS) ** 2
)


# Convert the dictionary to a list of keys and a list of the corresponding values
x = torch.tensor(list(corpus.keys()), dtype=torch.long)
degrees = torch.tensor(list(corpus.values()), dtype=torch.float)


# Define the model
w2v_kmer_model = Word2VecKMerEmb(kmer_num=4**k,
                                 lr=0.1, epoch_num=100, batch_size = len(x), verbose=True)
w2v_kmer_model.learn(x, degrees)

embs = w2v_kmer_model.get_emb()

# Convert sequences to embeddings
embs_sequences = []
for seq in combined_sequences:
    seq_emb = torch.zeros(size=(2, ), dtype=torch.float)
    for i in range(len(seq)):
        seq_emb += embs[seq[i]]

    seq_emb /= len(seq)
    embs_sequences.append(seq_emb.numpy())

embs_sequences = np.array(embs_sequences)

# Plot the sequences embeddings
plt.figure(figsize=(10, 10))

for color_idx, color in enumerate(['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray']):
    plt.scatter(
        embs_sequences[color_idx*1000:(color_idx+1)*1000, 0],
        embs_sequences[color_idx*1000:(color_idx+1)*1000, 1], s=1, c=color
    )

# plt.show()
plt.savefig(f"./figures/{filename}_w={window_size}.png")
'''

# # Plot the embeddings
# plt.figure(figsize=(10, 10))
# plt.scatter(embs[:, 0], embs[:, 1])
# for i in range(embs.shape[0]):
#     plt.annotate(i, (embs[i, 0], embs[i, 1]))
# plt.show()

#
# # Compute the accuracy
# with torch.no_grad():
#
#     log_softmax = kmer_emb_model.goforward(x=x)
#     predicted = torch.argmax(log_softmax, 1)
#
#     print(
#         "accuracy: ", (predicted == y).sum().item() / y.shape[0]
#     )


