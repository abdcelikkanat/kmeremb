import pickle as pkl
import time
import numpy as np
import itertools
import matplotlib.pyplot as plt

k = 2
dataset_sample_size = 1000

filename = f"zymohmw_samples={dataset_sample_size}_intkmerseq_k={k}_prob"  # f"3genomes_samples=1000_intkmerseq_k={k}"
input_file_path = f"./datasets/{filename}.pkl"

init_time = time.time()
with open(input_file_path, 'rb') as f:
    data = pkl.load(f)
sequences, probs = data['sequences'], data['probs']
print(f"- Time to load the sequences: {time.time() - init_time}")

# Convert the list of lists to a list
init_time = time.time()
probs = list(itertools.chain.from_iterable(probs))
print(f"- Time to convert the list of lists to a list: {time.time() - init_time}")

# Convert the list of lists to a list again
probs = list(itertools.chain.from_iterable(probs))
print(f"- Second step: time to convert the list of lists to a list: {time.time() - init_time}")


print(len(sequences))
# print(sequences[0])
print(probs[0])


# print(
#     len(probs)
# )
#
# hist, bin_edges = np.histogram(probs, bins=np.arange(0, 1.1, 0.1))
#
# print(
#     hist, bin_edges
# )
#
# print( hist / sum(hist) * 100 )
# plt.figure()
# plt.bar(bin_edges[:-1], hist, width=0.1)
# plt.show()

#
# print( np.arange(0, 1.1, 0.1) )



# print(
#     len(data['zymohmw'])
# )
#
# print(
#     data['zymohmw'][0][0].keys()
# )
#
# values = [value-33 for value in map(ord, data['zymohmw'][0][0]['ascii']) ]
#
# print(
# values, min(values), max(values)
# )