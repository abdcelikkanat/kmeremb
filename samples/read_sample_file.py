import sys
import pickle as pkl


file_path = sys.argv[1]

with open(file_path, 'rb') as f:
    data = pkl.load(f)

# 'geobacillus3', 'ecoli_k12_stat3', 'mruber1'

print(
    data['geobacillus3'][0].keys()
)