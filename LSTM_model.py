import torch
import torch.nn as nn
import torch.nn.functional as F


#Embedding module.
class Embed(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.W = nn.Parameter(torch.Tensor(vocab_size, embed_size))

    def forward(self, x):
        return self.W[x]

    def __repr__(self):
        return "Embedding(vocab: {}, embedding: {})".format(self.vocab_size, self.embed_size)

class Linear(nn.Module): #Linear layer
    def __init__(self, input_size, hidden_size):
        super().__init__() #input_size is the input size
        self.input_size = input_size #input_size is the input size
        self.hidden_size = hidden_size #hidden_size is the output size
        self.W = nn.Parameter(torch.Tensor(hidden_size, input_size)) #W is a matrix of size hidden_size x input_size
        self.b = nn.Parameter(torch.Tensor(hidden_size)) #b is a vector of size hidden_size

    def forward(self, x):
        z = torch.addmm(self.b, x.view(-1, x.size(2)), self.W.t()) #z = b + xW^T
        return z

    def __repr__(self):
        return "FC(input: {}, output: {})".format(self.input_size, self.hidden_size) #input_size is the input size, hidden_size is the output size

#The model as described in the paper
class LSTM_Model(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.vocab_size = params.vocab_size
        self.hidden_size = params.hidden_size
        self.layer_num = params.layer_num
        self.winit = params.winit 
        self.lstm_type = params.lstm_type
        self.embed = Embed(self.vocab_size, self.hidden_size)
        self.rnns = [nn.LSTM(self.hidden_size, self.hidden_size) for i in range(self.layer_num)]
        self.rnns = nn.ModuleList(self.rnns)
        self.fc = Linear(self.hidden_size, self.vocab_size) 
        self.dropout = nn.Dropout(p=params.dropout) 
        self.reset_parameters() 
        
    def reset_parameters(self):
        for param in self.parameters():
            nn.init.uniform_(param, -self.winit, self.winit)
            
    def state_init(self, batch_size):
        dev = next(self.parameters()).device
        states = [(torch.zeros(1, batch_size, layer.hidden_size, device = dev), torch.zeros(1, batch_size, layer.hidden_size, device = dev)) for layer in self.rnns]
        return states
    
    def detach(self, states):
        return [(h.detach(), c.detach()) for (h,c) in states]
    
    def forward(self, x, states):
        x = self.embed(x)
        x = self.dropout(x)
        for i, rnn in enumerate(self.rnns):
            x, states[i] = rnn(x, states[i])
            x = self.dropout(x)
        scores = self.fc(x)
        return scores, states
    
