import random
import torch

def getVocab():
  vocab = {}
  for i, word in enumerate(words):
    vocab[word] = i
  return vocab

def getVocabReverse():
  vocab = {}
  for i, word in enumerate(words):
    vocab[i] = word
  return vocab

def getData(batch, length):
  forward = random.choices(embedded_words, k=batch*length)
  reverse = forward.copy()
  reverse.reverse()
  
  return torch.LongTensor(forward), torch.LongTensor(reverse)



word_bank = 'in near total and have a better sense of smell, but poorer color vision. Cats, despite being solitary'
words = word_bank.split(' ')
words = list(set(words))
embed = getVocab()

embedded_words = [embed[i] for i in words]