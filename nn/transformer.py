import torch
import math
import os
import time

import torch.nn.functional as F


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        assert os.path.exists(path)
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

class TransformerModel(torch.nn.Transformer):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.position_encoder = PositionalEncoding(ninp,dropout)
        self.input_emb = torch.nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = torch.nn.Linear(ninp, ntoken)
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        torch.nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        torch.nn.init.zeros_(self.decoder.bias)
        torch.nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.position_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data

def get_batch(source, i):
    seq_len = min(35, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout=0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def EncoderLayer(src: torch.tensor, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
    encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
    return encoder_layer(src)

def train(model, ntokens, train_data, criterion, optimizer, bptt, lr, epoch):
    model.train()
    total_loss = 0.
    start_time = time.time()
    log_interval = 200

    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output =  model(data)
        output = output.view(-1, ntokens)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} '.format(
                epoch, batch, len(train_data) // bptt, lr,
                elapsed * 1000 / log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()

def evaluate(model, eval_data, ntokens, criterion, bptt):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            output = model(data)
            output = output.view(-1, ntokens)

            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(eval_data) - 1)

def main():
    nipn = 512
    nhead = 8
    nhid = 200
    nlayers = 2
    dropout = 0.2
    lr = 20
    bptt = 35
    corpus = Corpus("..\\data\\wikitext-2")
    train_data = batchify(corpus.train, bsz=20)
    eval_data = batchify(corpus.valid, bsz=10)
    #test_data = batchify(corpus.test, bsz=10)
    criterion = torch.nn.NLLLoss()
    ntokens = len(corpus.dictionary)
    model = TransformerModel(ntokens, nipn, nhead, nhid, nlayers, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr)

    best_val_loss = None
    for epoch in range(1, 2):
        epoch_start_time = time.time()
        train(model, ntokens, train_data, criterion, optimizer, bptt, lr, epoch)
        val_loss = evaluate(model, eval_data, ntokens, criterion, bptt)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        if not best_val_loss or val_loss < best_val_loss:
            best_val_loss = val_loss
        else:
            lr /= 4.0

if __name__ == "__main__":
    main()