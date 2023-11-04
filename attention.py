import torch 
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, d_model: int, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x



class Comp:
    def config(self):
        print("hello")


comp1 = Comp()

comp1.config()