import numpy as np
import matplotlib.pyplot as plt

seq = "AAACAAGTGTTTAAAACCTTGT"

# Implement the kmer spectrum and plot the histogram
k = 3
spectrum = []
for i in range(len(seq) - k + 1):
    spectrum.append(seq[i:i + k])

spectrum.sort()
spectrum = np.array(spectrum)
spectrum, counts = np.unique(spectrum, return_counts=True)
#plt.bar(spectrum, counts)
plt.hist(counts, bins=range(1, 10), density=True)
plt.show()
