import sys
import numpy as np
from math import log2

class ReadData():
    def __init__(self, infile):
        self.infile = infile
    
    def load_data(self):
        dataset = np.loadtxt(self.infile, dtype=int, delimiter='\t', skiprows=1)
        return dataset
    
    def extract_labels(self):
        dataset = self.load_data()
        labels = []
        for row in dataset:
            labels.append(row[-1])
        return labels

class Inspect():
    def __init__(self, infile):
        self.read_data = ReadData(infile)
        self.dataset = self.read_data.load_data()
        self.labels = self.read_data.extract_labels()

    def entropy(self):
        labels = self.labels
        counts = np.bincount(labels)
        probabilities = counts / len(labels)
        entropy = -sum(p * log2(p) for p in probabilities if p > 0)
        # entropy = format(entropy, '.6f')
        return entropy

    def error_rate(self):
        labels = self.labels
        counts = np.bincount(labels)
        error_rate = 1 - max(counts) / len(labels)
        # error_rate = format(error_rate, '.6f')
        return error_rate

if __name__ == "__main__":
    infile = sys.argv[1] # tsv file
    outfile = sys.argv[2] # txt file

    inspect = Inspect(infile)
    entropy = inspect.entropy()
    error_rate = inspect.error_rate()
    with open(outfile, "w") as file:
        file.write("entropy: " + str(entropy) + '\n')
        file.write("error: " + str(error_rate) + '\n')