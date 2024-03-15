import pickle as pkl


genome_name = 'geobacillus3'
k = 4
filename = "3genomes_samples=100"
input_file_path = f"./samples/{filename}.pkl"
output_file_path = f"./datasets/{filename}_data_k={k}.pkl"

with open(input_file_path, 'rb') as f:
    data = pkl.load(f)

reads = data[genome_name]
reads_num = len(reads)

samples = []
for i in range(reads_num):

    current_read = reads[i]['read'][0]
    ascii = list(map(ord, reads[i]['ascii']))

    if len(current_read) <= 2*k+1:
        continue

    for j in range(len(current_read) - 2*k):

        fragment = current_read[j:j+2*k+1]
        scores = ascii[j:j+2*k+1]

        samples.append( ( fragment[:k], fragment[k], fragment[k+1:], scores[:k], scores[k], scores[k+1:]) )


with open(output_file_path, 'wb') as f:
    pkl.dump(samples, f)

print(
    samples[0],
    samples[-1]
)