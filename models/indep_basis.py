import time
import pickle as pkl
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


if __name__ == "__main__":

    normalize = True
    dataset_sample_size = 1000
    num_processes = 1
    k = 2 #int(sys.argv[1]) #2
    batch_size = 0
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

    # Define the output file name
    output_filename = f"{filename}_pca_basis"

    # Construc the basis and sequences embeddings
    basis = np.eye(4**k, 4**k, dtype=float)

    read_embs = np.zeros(shape=(len(sequences), basis.shape[1]), dtype=float)
    for seq_idx, current_seq in enumerate(sequences):
        counts = np.bincount(current_seq, minlength=( 4** k ))
        read_embs[seq_idx] = np.dot(counts, basis)
        read_embs[seq_idx] /= len(current_seq)

    # Apply pca to get 2d embeddings
    pca = PCA(n_components=2)
    emd = pca.fit_transform(read_embs)

    # Plot the sequences embeddings
    plt.figure(figsize=(10, 10))

    for color_idx, color in enumerate(['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray']):
        plt.scatter(
            read_embs[color_idx * dataset_sample_size:(color_idx + 1) * dataset_sample_size, 0],
            read_embs[color_idx * dataset_sample_size:(color_idx + 1) * dataset_sample_size, 1], s=1, c=color
        )

    plt.savefig(f"./new_figures/{output_filename}.png")
