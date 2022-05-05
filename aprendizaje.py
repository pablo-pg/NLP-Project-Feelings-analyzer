#!/usr/bin/python
# -*- coding: utf-8 -*-
# Proyecto sobre Procesamiento de Lenguaje Natural para Inteligencia Artificial Avanzada
# Autor: Pablo Perez Gonzalez (alu0101318318@ull.edu.es)


import math
import pandas as pd
import re
import string

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from copy import deepcopy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

def nltk_pos_tagger(nltk_tag):
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:          
    return None




def read_and_tokenize(rawMessages):
  #{ Part-of-speech constants
  ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
  #}
  POS_LIST = [NOUN, VERB, ADJ, ADV]
  en_stops = set(stopwords.words('english'))
  punctuation_marks = string.punctuation + '…' + '”' + '“' + '-' + '‘'+ '’' + '´' + '—' + '`'
  data = list()
  for message in rawMessages :
    # Delete links
    message[0] = re.sub(r'http\S+', '', message[0])
    # Delete html tags
    message[0] = re.sub(r'<[^>]+>', '', message[0])
    # Delete emoji
    # message[0] = re.sub(r'(\uD83C[\uDDE6-\uDDFF]\uD83C[\uDDE6-\uDDFF])', '', message[0])
    # Delete usernames and hastags hastags
    message[0] = re.sub(r'(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)', '', message[0])
    # Every line will be tokenized into a word without taking into account the punctuation marks
    message[0] = message[0].translate(str.maketrans(dict.fromkeys(punctuation_marks, ' ')))
    
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(message[0]))
    wordnet_tagged = map(lambda x: (x[0], nltk_pos_tagger(x[1])), nltk_tagged)
    for word, tag in wordnet_tagged:
      if (word not in en_stops):
        if (word.isalpha()):
          word = word.lower()
          if tag is None:
            data.append(word)
          else:        
            data.append(lemmatizer.lemmatize(word, tag))
  return sorted(data)




def modelProcess(words, vocabulary, tweetsNumber, k):
  result = {
    'tweetsNumber': tweetsNumber,
    'wordsNumber': 0,
    'corpus': {}
  }

  for word in vocabulary:
    wordData = {'text': word, 'freq': 0, 'logProb': 0}
    result['corpus'][word] = wordData

  result['corpus']['__unknown__'] = {'text': '__unknown__', 'freq': 0, 'logProb': 0}
  wordsInMessages = 0

  for word in words:
    if word in result['corpus']:
      result['corpus'][word]['freq'] += 1
    else:
      result['corpus']['__unknown__']['freq'] += 1
    wordsInMessages += 1
  
  # print (wordsInMessages)
  result['wordsNumber'] = wordsInMessages

  for key in result['corpus']:
    # Suavizado Laplaciano
    result['corpus'][key]['freq'] += 1
    # Formula: P(palabra) = n_veces(palabra) + 1 / todas_palabras_corpus + n_palabras_vocabulario
    # En mi caso el +1 lo hago antes por lo que no tengo que sumar el 1
    result['corpus'][key]['logProb'] = math.log(result['corpus'][key]['freq'] / (wordsInMessages + len(vocabulary)))

  return result




def main():
  print("\nLoading...\n")
  df = pd.read_excel(r"COV_train.xlsx", sheet_name=0, dtype=str)
  mask = (df['Emotion'] == "Negative")
  negativeDf = df[mask]
  positiveDf = df[~mask]

  rawPositiveMessages = positiveDf[['Message']].to_numpy(dtype='str')
  rawNegativeMessages = negativeDf[['Message']].to_numpy(dtype='str')

  positiveWords = read_and_tokenize(rawPositiveMessages)
  negativeWords = read_and_tokenize(rawNegativeMessages)

  vocabulary = []
  with open('vocabulario.txt') as f:
    vocabulary = f.read().splitlines()

  # with open('vocabulario.txt') as f:
  #   vocabulary = f.readlines()
  
  vocabulary.pop(0)

  clonePos = deepcopy(vocabulary)
  cloneNeg = deepcopy(vocabulary)

  print("\nProcessing positive messages...\n")

  # k = 0 => Las palabras con 0 apariciones o menos se declararán como unknown
  positiveData = modelProcess(positiveWords, clonePos, len(rawPositiveMessages), 0)

  print("\nProcessing negative messages...\n")

  negativeData = modelProcess(negativeWords, cloneNeg, len(rawNegativeMessages), 0)

  positiveFilename = 'modelo_lenguaje_P.txt'
  negativeFilename = 'modelo_lenguaje_N.txt'

  with open(positiveFilename, 'w') as f:
    f.write('Numero de documentos (tweets) del corpus: %s\n' % positiveData['tweetsNumber'])
    f.write('Número de palabras del corpus: %s\n' % positiveData['wordsNumber'])
    for item in positiveData['corpus']:
      # print(positiveData['corpus'][item])
      f.write(f"Palabra: {positiveData['corpus'][item]['text']} Freq: {positiveData['corpus'][item]['freq']} LogProb: {positiveData['corpus'][item]['logProb']}\n")

  with open(negativeFilename, 'w') as f:
    f.write('Numero de documentos (tweets) del corpus :%s\n' % negativeData['tweetsNumber'])
    f.write('Número de palabras del corpus:%s\n' % negativeData['wordsNumber'])
    for item in negativeData['corpus']:
      f.write(f"Palabra: {negativeData['corpus'][item]['text']} Freq: {negativeData['corpus'][item]['freq']} LogProb: {negativeData['corpus'][item]['logProb']}\n")


main()

