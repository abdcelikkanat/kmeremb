import torch
import math
import time
import sys
import random
import pickle as pkl
import itertools
import matplotlib.pyplot as plt

class KMerEmb(torch.nn.Module):
    def __init__(self,  kmer_num, dim=128, dim1=64, dim2=32, dim3=16, dim4=4,
                 lr=0.1, epoch_num=100, batch_size = 1000,
                 device=torch.device("cpu"), verbose=False, seed=0):

        super(KMerEmb, self).__init__()

        self.__seed = seed
        self.__kmer_num = kmer_num
        self.__dim = dim
        self.__dim1, self.__dim2, self.__dim3, self.__dim4 = dim1, dim2, dim3, dim4
        self.__lr = lr
        self.__epoch_num = epoch_num
        self.__batch_size = batch_size
        self.__device = device
        self.__verbose = verbose

        self.__set_seed(seed)

        self.__embs = torch.nn.Parameter(
            2 * torch.rand(size=(self.__kmer_num, self.__dim), device=self.__device) - 1, requires_grad=True
        )

        self.__h0 = torch.nn.Parameter(
            2 * torch.rand(size=(2*self.__dim, self.__dim1), device=self.__device) - 1, requires_grad=True
        )
        self.__bias0 = torch.nn.Parameter(
            2 * torch.rand(size=(self.__dim1, ), device=self.__device) - 1, requires_grad=True
        )
        self.__h1 = torch.nn.Parameter(
            2 * torch.rand(size=(self.__dim1, self.__dim2), device=self.__device) - 1, requires_grad=True
        )
        self.__bias1 = torch.nn.Parameter(
            2 * torch.rand(size=(self.__dim2,), device=self.__device) - 1, requires_grad=True
        )
        self.__h2 = torch.nn.Parameter(
            2 * torch.rand(size=(self.__dim2, self.__dim3), device=self.__device) - 1, requires_grad=True
        )
        self.__bias2 = torch.nn.Parameter(
            2 * torch.rand(size=(self.__dim3,), device=self.__device) - 1, requires_grad=True
        )
        self.__h3 = torch.nn.Parameter(
            2 * torch.rand(size=(self.__dim3, self.__dim4), device=self.__device) - 1, requires_grad=True
        )
        self.__bias3 = torch.nn.Parameter(
            2 * torch.rand(size=(self.__dim4,), device=self.__device) - 1, requires_grad=True
        )

        self.__optimizer = torch.optim.Adam(self.parameters(), lr=self.__lr)
        self.__loss = []

        # self.__pdist = torch.nn.PairwiseDistance(p=2)

    def __set_seed(self, seed=None):

        if seed is not None:
            self._seed = seed

        random.seed(self._seed)
        torch.manual_seed(self._seed)

    def goforward(self, x):

        layer0 = torch.hstack((
            torch.index_select(self.__embs, 0, x[:, 0]), torch.index_select(self.__embs, 0, x[:, 1])
        ))  # batch_size x (2 x dim)

        layer1 = layer0 @ self.__h0 #+ self.__bias0.unsqueeze(0)  # batch_size x dim1
        layer1 = torch.special.expit(layer1)
        layer2 = layer1 @ self.__h1 #+ self.__bias1.unsqueeze(0)  # batch_size x dim2
        layer2 = torch.special.expit(layer2)
        layer3 = layer2 @ self.__h2 #+ self.__bias2.unsqueeze(0)  # batch_size x dim3
        layer3 = torch.special.expit(layer3)
        layer4 = layer3 @ self.__h3 #+ self.__bias3.unsqueeze(0)  # batch_size x dim4
        # layer4 = torch.special.expit(layer4)

        return torch.log_softmax(layer4, dim=1)

    def __compute_loss(self, x, y):

        log_softmax = self.goforward(x=x)

        return -torch.gather(log_softmax, 1, y.unsqueeze(1)).sum()

    def learn(self, x, y):

        for epoch in range(self.__epoch_num):

            # Shuffle data
            randperm = torch.randperm(x.shape[0])

            x = x[randperm]
            y = y[randperm]

            epoch_loss = 0
            for i in range(0, x.shape[0], self.__batch_size):

                batch_x = x[i:i+self.__batch_size]
                batch_y = y[i:i+self.__batch_size]

                if batch_x.shape[0] != self.__batch_size:
                    continue

                self.__optimizer.zero_grad()

                batch_loss = self.__compute_loss(batch_x, batch_y)
                batch_loss.backward()
                self.__optimizer.step()

                self.__optimizer.zero_grad()

                epoch_loss += batch_loss.item()

            epoch_loss /= math.ceil(x.shape[0] / self.__batch_size)

            if self.__verbose:
                print(f"epoch: {epoch}, loss: {epoch_loss}")

            self.__loss.append(epoch_loss)


class Dataset:

    def __init__(self, file_path, k):

        self.__letters = ['A', 'C', 'G', 'T']
        self.__kmers = [ ''.join(kmer) for kmer in itertools.product(self.__letters, repeat=k) ]
        self.__letter2id = {letter: idx for idx, letter in enumerate(self.__letters)}
        self.__kmer2id = {kmer: idx for idx, kmer in enumerate(self.__kmers)}


        with open(file_path, 'rb') as f:
            data = pkl.load(f)

        self.__output = []
        for sample in data:
            self.__output .append(
                (
                    self.__kmer2id[sample[0]],
                    self.__kmer2id[sample[2]],
                    self.__letter2id[sample[1]],
                )
            )

    def get_data(self):

        data = torch.as_tensor(self.__output, dtype=torch.long)

        return data[:, :-1], data[:, -1]

k = 4
input_file_path = f"./datasets/3genomes_samples=100_data_k={k}.pkl"
dataset = Dataset(input_file_path, k=k)
x, y = dataset.get_data()

kmer_emb_model = KMerEmb(kmer_num=4**k, lr=0.1, epoch_num=100, batch_size = 10000, verbose=True)
kmer_emb_model.learn(x, y)

# Compute the accuracy
with torch.no_grad():

    log_softmax = kmer_emb_model.goforward(x=x)
    predicted = torch.argmax(log_softmax, 1)

    print(
        "accuracy: ", (predicted == y).sum().item() / y.shape[0]
    )



# filename = "3genomes_samples=100"
# k = 4
#
# file_path = f"./datasets/{filename}_data_k={k}.pkl"
# with open(file_path, 'rb') as f:
#     data = pkl.load(f)
#
# reads = data['geobacillus3']
# reads_num = len(reads)
#
# max_value = -1
# scores = []
# for i in range(reads_num):
#
#     max_value = max(
#         max_value, max(map(ord, reads[i]['ascii']))
#     )
#
#     scores.extend(
#         list(map(ord, reads[i]['ascii']))
#     )
#
#
# # 'geobacillus3', 'ecoli_k12_stat3', 'mruber1'
# # 'id', 'mm', 'ml', 'read', 'ascii'
# print(
#     "max value: ", max_value
# )
#
# plt.figure(figsize=(10, 5))
# plt.hist(scores, bins=range(0, 100))
# plt.show()