from __future__ import division
import torch
import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F
import torch.optim as optim
from dataprep import *

torch.manual_seed(1)

class embed(torch.nn.Module):
  def __init__(self, max_length, vocab_length, embedding_dim):
    super(embed, self).__init__()
    self.embeds = Embedding(vocab_length, embedding_dim,  sparse=False)
    self.embedding_dim = embedding_dim
    self.max_length = max_length
    self.batch_size = batch_size
  
  def forward(self, x):
    return self.embeds(x).view(-1, self.max_length, self.embedding_dim)

class scaledDotProductAttention(torch.nn.Module):
  def __init__(self):
    super(scaledDotProductAttention, self).__init__()
    self.softmax = Softmax()

  def forward(self, q, k, v, dk):
    out = torch.div(q.bmm(torch.transpose(k, 2, 1)), dk**0.5).bmm(v)
    return self.softmax(out)

class multiheadAttention(torch.nn.Module):
  def __init__(self, model_dim, h):
    super(multiheadAttention, self).__init__()
    self.model_dim = model_dim
    self.h = h
    self.dk = model_dim // h
    self.dv = model_dim // h
    self.scaledDotProductAttention = scaledDotProductAttention()
    self.wq = nn.ModuleList()
    self.wk = nn.ModuleList()
    self.wv = nn.ModuleList()
    self.wo = Linear(h * self.dv, model_dim)

    for _ in range(self.h):
      self.wq.append(Linear(self.model_dim, self.dk)) 
      self.wk.append(Linear(self.model_dim, self.dk)) 
      self.wv.append(Linear(self.model_dim, self.dv))

  def forward(self, q, k, v):
    heads = []
    # We can do this in parallel
    for i in range(self.h):
      heads.append(self.scaledDotProductAttention(self.wq[i](q), self.wk[i](k), self.wv[i](v), self.dk))
    heads = torch.cat(heads, 2)
    return self.wo(heads)

class transformer(torch.nn.Module):
  def __init__(self, batch_size, model_dim, n, h, max_length, vocab_size):
    super(transformer, self).__init__()
    self.model_dim = model_dim
    self.max_length = max_length
    self.n = n
    self.h = h
    self.batch_size = batch_size
    self.encode_att = nn.ModuleList([multiheadAttention(model_dim, h) for _ in range(self.n)])
    self.encode_linear = nn.ModuleList([Sequential(Linear(model_dim, model_dim), ReLU(), Linear(model_dim, model_dim), ReLU()) for _ in range(self.n)])
    self.decode_att_masked = nn.ModuleList([multiheadAttention(model_dim, h) for _ in range(self.n)])
    self.decode_att = nn.ModuleList([multiheadAttention(model_dim, h) for _ in range(self.n)])
    self.decode_linear = nn.ModuleList([Sequential(Linear(model_dim, model_dim), ReLU(), Linear(model_dim, model_dim), ReLU()) for _ in range(self.n)])
    self.softmax = Softmax()
    self.dropout = torch.nn.Dropout(p=0.1)
    self.layer_norm = LayerNorm([self.max_length, self.model_dim])
    
    self.final_linear = Linear(model_dim, vocab_size)

  def decode(self, outputs, pos, encoder_outputs):
    for i in range(self.n):
      identity = outputs
      out = self.layer_norm(self.dropout(self.decode_att_masked[i](outputs, outputs, outputs)) + identity)
      
      identity = out
      out = self.layer_norm(self.dropout(self.decode_att[i](encoder_outputs, outputs, outputs)) + identity)

      identity = out
      out = self.layer_norm(self.dropout(self.decode_linear[i](out)) + identity)
      
    outputs = self.final_linear(out)
    return outputs[:, pos]


  def forward(self, x):
    identity = x
    start_token = torch.zeros(self.batch_size, self.max_length, self.model_dim, requires_grad=False)
    outs = []
    outputs = start_token
    for i in range(self.n):

      out = self.layer_norm(self.encode_att[i](x, x, x) + identity)

      identity = out
      out = self.layer_norm(self.dropout(self.encode_linear[i](out)) + identity)
      x = out

    pos = 1
    next_element = self.decode(outputs, pos, x)
    torch.cat((outputs, next_element.view(64, -1, 1)), 2)
    while next_element.item() != 0:
      next_element = self.decode(outputs, pos)
      torch.cat((outputs, next_element), 0)
      print(outputs)

    return outputs


max_length = 10
batch_size = 64
model_dim = 128

vocab = getVocab()
unembed = getVocabReverse()
vocab_length = len(vocab)
embedder = embed(max_length, vocab_length, model_dim)
t = transformer(batch_size, model_dim, 8, 8, max_length, vocab_length)
criterion = CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(t.parameters(), lr=3e-4)

# Train
total_loss = torch.Tensor(0)
batch = 100

for i in range(10001):
  data, target = getData(batch_size, max_length)

  pred = t(embedder(data))
  values, indexes = pred.max(2)

  loss = criterion(torch.transpose(pred, 1, 2), target.view(batch_size, max_length)) / batch_size
  print(loss.item())
  #if (i % batch == 0):
  #  predictions = []
  #  for index in indexes:
  #    predictions.append(unembed[index.item()])
  #  print('target', target)
  #  print('total loss', total_loss)
  #  print('predictions', predictions)
  #  print('softmax', pred)
  #  print(pred.max(0), pred.max(1))

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()


  
