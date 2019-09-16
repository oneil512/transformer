import random

word_bank = 'in near total and have a better sense of smell, but poorer color vision. Cats, despite being solitary'
words = word_bank.split(' ')
words = list(set(words))

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

def getData(n):
  random.shuffle(words)
  return words[:n], words[:n][::-1]


