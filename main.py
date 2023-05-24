import torch.nn as nn
import torch
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader
from typing import List, Tuple, Any
from torch.utils.tensorboard import SummaryWriter
import itertools
from pathlib import Path
from torch.optim.lr_scheduler import StepLR
from LSTM_model import LSTM_Model

# set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#TODO rename word_log_probs to output
#TODO the model is unrolled 35 times
#TODO (successive minibatches sequentially traverse the training set
#TODO its parameters are initialized uniformly in [−0.05, 0.05]
#TODO should I remove the <unk> token? and the , and .?

# TODO PARAMETERS TO PLAY WITH - LEARNING RATE, EMBEDDING SIZE, HIDDEN SIZE 

params = lambda: None # create an empty object
params.lstm_type = 'pytorch' # type of LSTM
params.layer_num = 2 # number of layers
params.hidden_size = 200 # hidden size
params.batch_size = 20 # batch size
params.learning_rate = 0.8 # learning rate
params.epochs = 20 # number of epochs
params.epoch_threshold = 6 # epoch threshold
params.factor = 1.5 # factor
params.num_batches = 10000 # number of batches
params.dropout = 0.5 # dropout probability
params.seq_length = 35 # sequence length
params.clip_grad_value = 5 # clip gradient value
params.winit = 0.05 # weight initialization

SAVED_MODELS_DIR = 'saved_models' # directory to save models

def data_read():
    with open("PTB/ptb.train.txt") as f:
        file = f.read() # read the file
        train = file[1:].split(' ') # remove the first space and split the file into words
    with open("PTB/ptb.valid.txt") as f:
        file = f.read() # read the file
        valid = file[1:].split(' ') # remove the first space and split the file into words
    with open("PTB/ptb.test.txt") as f:
        file = f.read() # read the file
        test = file[1:].split(' ') # remove the first space and split the file into words
    words = sorted(set(train)) # get the unique words
    params.vocab_size = len(words) # get the vocab size
    char2ind = {c: i for i, c in enumerate(words)} # create a mapping between characters and indices
    train = [char2ind[c] for c in train] # convert the characters to indices
    valid = [char2ind[c] for c in valid] # convert the characters to indices
    test = [char2ind[c] for c in test] # convert the characters to indices
    return np.array(train).reshape(-1, 1), np.array(valid).reshape(-1, 1), np.array(test).reshape(-1, 1), len(words)

def batchify(data, batch_size, seq_length):
    data = torch.tensor(data, dtype = torch.int64) # convert to tensor
    num_batches = data.shape[0] // (batch_size) # number of batches
    data = data[:num_batches * batch_size] # cut off the end so that it divides evenly
    data = data.view(batch_size, -1) # reshape into batch_size rows
    datatoplot = data[:5, :seq_length*10] # take a subset for plotting
    print(datatoplot) # print it out
    x_y_pairs = [] # initialize the list of x, y pairs
    for i in range(0, data.shape[1] - seq_length, seq_length): # loop through the data
        x = data[:, i:i+seq_length].T # get the input
        y = data[:, i+1:i+seq_length+1].T # get the target
        x_y_pairs.append((x, y)) # append to the list
    return x_y_pairs # return the list

def train_model(model, train, valid, test, params):
    total_words = 0
    lr = params.learning_rate
    print("Starting training.\n")
    for epoch in range(params.epochs):
        states = model.state_init(params.batch_size)
        model.train()
        if epoch > params.epoch_threshold:
            lr = lr / params.factor
        for i, (x, y) in enumerate(train):
            total_words += x.numel()
            model.zero_grad()
            states = model.detach(states)
            scores, states = model(x, states)
            loss = nll_loss(scores, y)
            loss.backward()
            with torch.no_grad():
                norm = nn.utils.clip_grad_norm_(model.parameters(), params.clip_grad_value)
                for w_parameters in model.parameters():
                    w_parameters -= lr * w_parameters.grad
            if i % (len(train)//10) == 0:
                print("batch no = {:d} / {:d}, ".format(i, len(train)) +
                    "train loss = {:.3f}, ".format(loss.item()/params.batch_size) +
                    "dw.norm() = {:.3f}, ".format(norm) +
                    "lr = {:.3f}, ".format(lr))
        model.eval()
        valid_perp = perplexity(valid, model, params)
        print("Epoch : {:d} || Validation set perplexity : {:.3f}".format(epoch+1, valid_perp))
        print("*************************************************\n")
        test_perp = perplexity(test, model, params)
        print("Test set perplexity : {:.3f}".format(test_perp))
    print("Training is over.")
    return model

def nll_loss(scores, y):
    batch_size = y.size(1)
    expscores = scores.exp()
    probabilities = expscores / expscores.sum(1, keepdim = True)
    answerprobs = probabilities[range(len(y.reshape(-1))), y.reshape(-1)]
    #I multiply by batch_size as in the original paper
    #Zaremba et al. sum the loss over batches but average these over time.
    return torch.mean(-torch.log(answerprobs) * batch_size)

def perplexity(data, model, params):
    with torch.no_grad():
        losses = []
        states = model.state_init(params.batch_size)
        for x, y in data:
            scores, states = model(x, states)
            loss = nll_loss(scores, y)
            #Again with the sum/average implementation described in 'nll_loss'.
            losses.append(loss.data.item()/params.batch_size)
    return np.exp(np.mean(losses))


def main():
    # The vocab is built from the training data
    # If a word is missing from the training data, it will be replaced with <unk>
    train, valid, test, vocab_size = data_read()
    params.vocab_size = vocab_size
    train_batchwise_x_y_pairs = batchify(train, params.batch_size, params.seq_length)
    valid_batchwise_x_y_pairs = batchify(valid, params.batch_size, params.seq_length)
    test_batchwise_x_y_pairs = batchify(test, params.batch_size, params.seq_length)
    lstm_model = LSTM_Model(params)
    lstm_model.to(device)
    lstm_model = train_model(lstm_model,train_batchwise_x_y_pairs, valid_batchwise_x_y_pairs, test_batchwise_x_y_pairs, params)
    
    #TODO add perplexity calculation
    #TODO add tensorboard
    #TODO add saving the model
    #TODO add loading the model
    #TODO add testing the model
    #TODO add hyperparameter tuning
    #TODO add dropout
    #TODO add batch normalization

if __name__ == "__main__":
    main()


