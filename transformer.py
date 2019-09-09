from __future__ import division
import torch
from torch.nn import *
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class embed(torch.nn.Module):
  def __init__(self, vocab, embedding_dim):
    super(embed, self).__init__()
    # {"hello": 0, "world": 1}
    self.word_to_ix = vocab
    self.embeds = Embedding(len(vocab), embedding_dim)
    self.embedding_dim = embedding_dim
  
  def forward(self, x):
    xs = x.split(" ")
    vectors = []
    for word in xs:
      lookup = torch.tensor([self.word_to_ix[word]], dtype=torch.long)
      vectors.append(self.embeds(lookup))
    return torch.stack(vectors, 0).view(-1, self.embedding_dim)

  #def getPositionalEmbeddings(x):
      
    

class scaledDotProductAttention(torch.nn.Module):
  def __init__(self):
    super(scaledDotProductAttention, self).__init__()
    self.softmax = Softmax()

  def forward(self, q, k, v, dk):
    out = torch.div(q.mm(k.t()), dk**0.5).mm(v)
    return self.softmax(out)

class multiheadAttention(torch.nn.Module):
  def __init__(self, model_dim, h):
    super(multiheadAttention, self).__init__()
    self.model_dim = model_dim
    self.h = h
    self.dk = model_dim // h
    self.dv = model_dim // h
    self.scaledDotProductAttention = scaledDotProductAttention()
    self.wq = []
    self.wk = []
    self.wv = []
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
    heads = torch.cat(heads, 1)
    return self.wo(heads)

class transformer(torch.nn.Module):
  def __init__(self, model_dim, n, h, max_length, vocab_size):
    super(transformer, self).__init__()
    self.model_dim = model_dim
    self.max_length = max_length
    self.n = n
    self.h = h
    self.encode_att = []
    self.encode_linear = []
    self.decode_att_masked = []
    self.decode_att = []
    self.decode_linear = []
    self.softmax = Softmax()
    self.layer_norm = LayerNorm([self.max_length, self.model_dim])
    
    for _ in range(n):
      self.encode_att.append(multiheadAttention(model_dim, h))
      self.encode_linear.append(Sequential(Linear(model_dim, model_dim), ReLU(), Linear(model_dim, model_dim), ReLU())) 

      self.decode_att_masked.append(multiheadAttention(model_dim, h))
      self.decode_att.append(multiheadAttention(model_dim, h))
      self.decode_linear.append(Sequential(Linear(model_dim, model_dim), ReLU(), Linear(model_dim, model_dim), ReLU()))

    self.final_linear = Linear(self.n * model_dim, vocab_size)

  def forward(self, x):
    identity = x
    start_token = torch.zeros(self.max_length, self.model_dim, requires_grad=False)
    outs = []
    outputs = start_token
    # Should do in parallel
    for i in range(self.n):

      out = self.layer_norm(self.encode_att[i](x, x, x) + identity)

      identity = out
      out = self.layer_norm(self.encode_linear[i](out) + identity)
      
      identity = outputs
      d_out = self.layer_norm(self.decode_att_masked[i](outputs, outputs, outputs) + identity)
      
      identity = d_out
      out = self.layer_norm(self.decode_att[i](d_out, out, out) + identity)
  
      identity = out
      out = self.layer_norm(self.decode_linear[i](out) + identity)
      outs.append(out)
    outputs = torch.cat(outs, 1)
    out = self.final_linear(outputs)
    return self.softmax(out)

max_length = 10
model_dim = 20

vocab = {'the': 0, 'cat': 1}
unembed = {0: 'the', 1: 'cat'}
vocab_size = len(vocab)
embedder = embed(vocab, model_dim)
t = transformer(model_dim, 4, 4, max_length, vocab_size)
criterion = CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(t.parameters(), lr=1e-4)
onehot = torch.FloatTensor(10, vocab_size)

# Train
for i in range(1000):
  s='the cat the cat the the cat the cat the'
  words = s.split(' ')
  target = []

  for word in words:
    target.append(vocab[word])
  target = torch.tensor(target)
  onehot.zero_()
  onehot_targets = onehot.scatter_(1, target.view(-1, 1), 1).long()

  pred = t(embedder(s))
  values, indexes = pred.max(1)
  #onehot.zero_()
  #onehot_indexes = onehot.scatter_(1, indexes.view(-1, 1), 1)

  
  loss = criterion(pred, target)
  print(loss.item())
  print(i)
  if (i % 100 == 0):
    predictions = []
    for index in indexes:
      predictions.append(unembed[index.item()])
    print(predictions)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()