import torch
import torch.nn as nn
import torch.nn.functional as F


#The model as described in the paper
class LSTM_Model(nn.Module):
    def __init__(self, params, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = params.hidden_size
        self.layer_num = params.layer_num
        self.winit = params.winit 
        self.lstm_type = params.lstm_type
        self.embed = nn.Embedding(self.vocab_size, self.hidden_size) 
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, dropout = params.dropout, num_layers = self.layer_num, batch_first = True)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        self.dropout = nn.Dropout(p=params.dropout) 
        self.reset_parameters() 
        
    def reset_parameters(self):
        for param in self.parameters():
            nn.init.uniform_(param, -self.winit, self.winit)
            
    def detach(self, hidden_states):
        if type(hidden_states) == tuple:
            hidden_states = (hidden_states[0].detach(), hidden_states[1].detach())
        else:
            hidden_states = hidden_states.detach()
        return hidden_states
    

    def forward(self, x, states):
        x = self.embed(x)
        x = self.dropout(x)
        x = self.rnn(x, states)
        scores = self.fc(x)
        return scores, states
    
