import sys
import numpy as np
from math import log2

# Read data

class ReadData():
    def __init__(self, infile):
        self.infile = infile
    
    def load_data(self):
        dataset = np.loadtxt(self.infile, dtype=int, delimiter='\t', skiprows=1)
        return dataset
    
class MajorityVote():
    def __init__(self, infile, outfile):
        self.read_data = ReadData(infile)
        self.dataset = self.read_data.load_data()
        self.labels = self.read_data.extract_labels()
        self.features = self.read_data.extract_features()
        self.outfile = outfile

    def train(self):
        dataset = self.dataset
        vote = self.majority_vote(dataset)
        return vote

    def majority_vote(self, dataset):
        count1, count0 = 0, 0
        for row in dataset:
            if row[-1] == 1:
                count1 += 1
            else:
                count0 += 1
        if count1 >= count0: return 1
        else:                return 0

    # Predict on train data

    def predict(self, best_vote):
        predict_label = []
        for row in self.dataset:
            predict_label.append(best_vote)
        self.predict_label = np.array(predict_label)
        np.savetxt(self.outfile, predict_label, fmt='%i', delimiter='\n')
        return predict_label

    # Compute train error

    def error(self):
        error_count = 0
        for row_index in range(len(self.dataset)):
            if self.dataset[row_index][-1] != self.predict_label[row_index]:
                error_count += 1
        error_rate = error_count / len(self.dataset)
        error_rate = format(error_rate, '.6f')
        return error_rate
    
if __name__ == "__main__":
    train_infile = sys.argv[1] # tsv file
    test_infile = sys.argv[2] # tsv file
    train_outfile = sys.argv[3] # txt file
    test_outfile = sys.argv[4] # txt file
    metrics = sys.argv[5] # txt file

    train = MajorityVote(train_infile, train_outfile)
    train_train = train.train()
    train_predict = train.predict(train_train)
    train_error = train.error()

    test = MajorityVote(test_infile, test_outfile)
    test_train = test.train()
    test_predict = test.predict(train_train)
    test_error = test.error()

    f = open(metrics, "w")
    f.write(f"error(train): {train_error}")
    f.write("\n")
    f.write(f"error(test): {test_error}")
    

