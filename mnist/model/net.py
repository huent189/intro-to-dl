import torch.nn as nn
import torch.nn.functional as F
import numpy as np
INPUT_DIM = 28 *28
OUTPUT_DIM = 10
loss = nn.CrossEntropyLoss()
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(INPUT_DIM, OUTPUT_DIM)

    def forward(self, x):
        x = x.view(-1, INPUT_DIM)
        return F.softmax(self.linear1(x))

def loss_fn(outputs, labels):
    return loss(outputs, labels)

def accuracy_fn(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(labels.size)

metrics = {
    'accuracy' : accuracy_fn,
}