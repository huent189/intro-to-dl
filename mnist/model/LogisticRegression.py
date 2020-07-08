import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.softmax(self.linear1(x))

def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss(outputs, labels)

def accuracy_fn(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(labels.size)

metrics = {
    'accuracy' : accuracy_fn,
}