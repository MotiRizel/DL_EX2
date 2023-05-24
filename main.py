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

def load_data():
    train_data = open('PTB/ptb.train.txt', 'r').read()
    valid_data = open('PTB/ptb.valid.txt', 'r').read()
    test_data = open('PTB/ptb.test.txt', 'r').read()
    return preprocess_data(train_data), preprocess_data(valid_data), preprocess_data(test_data) 

# define tokenizer and build vocabulary

def preprocess_data(text_data: str):
    """ Remove newlines and replace them with <eos> token """
    # Not sure if I should do this
    text_data = text_data.lower()
    text_data = text_data.replace('\r\n', '<eos>')
    text_data = text_data.replace('\n', '<eos>')
    return text_data

def tokenize(text_data: str):
    """
    Tokenize the data
    """
    return text_data.split()

def build_vocab(text_data: str):
    """
    Build a vocabulary from the data
    ie: mapping between a word and an index
    """
    words = tokenize(text_data)
    counter = Counter()
    counter.update(words)
    #TODO not sure - start from 1, 0 is reserved for padding - when I tried it it failed on some Error
    print("The total number of words is {}".format(len(words)))
    print("found {} unique tokens in training data".format(len(counter)))
    print(f"The 30 most common words are {counter.most_common(30)}")
    
    return {word: index for index, word in enumerate(counter.keys())}

class PennTreeBankDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data: str, vocab: dict, seq_length: int):
        self.raw_data = raw_data
        self.vocab = vocab
        self.encoded_text = self.text_to_vocab(raw_data, vocab)
        self.seq_length = seq_length
    
    def text_to_vocab(self, text: str, vocab: dict):
        """
        Convert the data to a vector of indices
        """
        return [vocab[word] for word in tokenize(text)]

    def __getitem__(self, index):
        """
        get the sentence at the index, then shift it by one word to the right to get the target sentence
        """
        return torch.tensor(self.encoded_text[index * self.seq_length:(index + 1) * self.seq_length] ,dtype=torch.long, device=device), \
            torch.tensor(self.encoded_text[index * self.seq_length + 1:(index + 1) * self.seq_length+1] ,dtype=torch.long, device=device)
        
    def __len__(self):
        return int(len(self.encoded_text) / self.seq_length) - 30


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    
    sentences = [item[0] for item in batch]
    target_sentences = [item[1] for item in batch]
    # transform the list of tensors to a tensor of tensors without using pad_sequence
    sentences = torch.stack(sentences, dim=0)
    target_sentences = torch.stack(target_sentences, dim=0)
    return sentences, target_sentences

def create_data_loaders(train_data: str, validation_data: str, test_data: str, batch_size: int, vocab: dict):
    train_loader = DataLoader(PennTreeBankDataset(train_data, vocab, params.seq_length), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(PennTreeBankDataset(validation_data, vocab, params.seq_length), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(PennTreeBankDataset(test_data, vocab, params.seq_length), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader

def train_model(model, train, valid, test, params):
    total_words = 0
    lr = params.learning_rate
    print("Starting training.\n")
    for epoch in range(params.epochs):
        states = model.state_init(params.batch_size)
        model.train()
        if epoch > params.epoch_threshold:
            lr = lr / params.factor
        for batch_number, (sentence, target_sentence) in enumerate(train):
            total_words += sentence.numel()
            model.zero_grad()
            states = model.detach(states)
            scores, states = model(sentence, states)
            loss = nll_loss(scores, target_sentence)
            loss.backward()
            with torch.no_grad():
                norm = nn.utils.clip_grad_norm_(model.parameters(), params.clip_grad_value)
                for w_parameters in model.parameters():
                    w_parameters -= lr * w_parameters.grad
            if batch_number % (len(train)//10) == 0:
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
    train_data, valid_data, test_data = load_data()
    vocab = build_vocab(train_data)
    train_loader, valid_loader, test_loader = create_data_loaders(train_data, valid_data, test_data, params.batch_size, vocab)

    lstm_model = LSTM_Model(params, len(vocab))
    lstm_model.to(device)
    lstm_model = train_model(lstm_model,train_loader, valid_loader, test_loader, params)
    
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


