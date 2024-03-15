import pickle as pkl
import sys
import time
import itertools
import matplotlib.pyplot as plt
import numpy as np

normalize = True
dataset_sample_size = 1000
num_processes = 1
k = 3 # int(sys.argv[1]) #2
lr = 0.01
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
        f"./embs/3___{filename}_{'' if normalize else 'un'}normalized_w={window_size}_"
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

# # Plot the sequences embeddings
# plt.figure(figsize=(10, 10))
# for color_idx, color in enumerate(['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray']):
#     plt.scatter(
#         read_embs[color_idx * dataset_sample_size:(color_idx + 1) * dataset_sample_size, 0],
#         read_embs[color_idx * dataset_sample_size:(color_idx + 1) * dataset_sample_size, 1], s=1, c=color
#     )
# plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans

N = len(labels)
_score_types = ['micro', 'macro']
_training_ratios = [0.1, 0.5, 0.9]
_number_of_shuffles = 1
K = class_num
results = {}

# for score_t in _score_types:
#             results[score_t] = OrderedDict()
#             for ratio in _training_ratios:
#                 results[score_t].update({ratio: []})
#
# for train_ratio in _training_ratios:
#
#     for _ in range(_number_of_shuffles):
#
#         read_embs, labels = shuffle(read_embs, labels)
#
#
#         # Get the training size
#         train_size = int(train_ratio * N)
#         # Divide the data into the training and test sets
#         train_features = read_embs[0:train_size, :]
#         train_labels = labels[0:train_size]
#
#         test_features = read_embs[train_size:, :]
#         test_labels = labels[train_size:]
#
#
#         # Train the classifier
#         ovr = OneVsRestClassifier(LogisticRegression(solver='liblinear'))
#
#         ovr.fit(train_features, train_labels)
#         # Find the predictions, each node can have multiple labels
#         test_prob = np.asarray(ovr.predict_proba(test_features))
#         y_pred = []
#         for i in range(len(test_labels)):
#             k = 1  # The number of labels to be predicted
#             pred = test_prob[i, :].argsort()[-k:]
#             y_pred.append(pred)
#
#         # Find the true labels
#         y_true = [[j] for j in test_labels]
#
#         mlb = MultiLabelBinarizer(classes=range(K))
#         for score_t in _score_types:
#             score = f1_score(y_true=mlb.fit_transform(y_true),
#                              y_pred=mlb.fit_transform(y_pred),
#                              average=score_t)
#
#             results[score_t][train_ratio].append(score)
#
# print(results)

kmeans = KMeans(n_clusters=K, random_state=0, n_init=100, init="k-means++").fit(read_embs)
# ovr = OneVsRestClassifier(kmeans).fit(read_embs)

y_true = [[l] for l in labels]
y_pred = [[l] for l in kmeans.labels_]  #ovr.predict(read_embs)

for score_t in _score_types:
    mlb = MultiLabelBinarizer(classes=range(K))
    score = f1_score(
        y_true=mlb.fit_transform(y_true), y_pred=mlb.fit_transform(y_pred), average=score_t
    )
    print(f"{score_t} score: {score}")


# Plot the sequences embeddings
plt.figure(figsize=(10, 10))
for color_idx, color in enumerate(['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray']):
    plt.scatter(
        read_embs[color_idx * dataset_sample_size:(color_idx + 1) * dataset_sample_size, 0],
        read_embs[color_idx * dataset_sample_size:(color_idx + 1) * dataset_sample_size, 1], s=1, c=color
    )
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='black', marker='x')
plt.show()