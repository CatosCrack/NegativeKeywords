# Experiment classifier
# Define custom classifier
import torch.nn as nn

class CustomSequential(nn.Sequential):
    def forward(self, input):
      input = input[:,0,:] # Extract CLS token, index 0
      for i, module in enumerate(self):
        input = module(input)
      return input
