#!/usr/bin/python
# -*- coding: utf-8 -*-
# Proyecto sobre Procesamiento de Lenguaje Natural para Inteligencia Artificial Avanzada
# Autor: Pablo Perez Gonzalez (alu0101318318@ull.edu.es)

import math
import pandas as pd
import re
import nltk
import string

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

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
    parsedMessage = ""
    # Delete links
    message[0] = re.sub(r'http\S+', '', message[0])
    # Delete html tags
    message[0] = re.sub(r'<[^>]+>', '', message[0])
    # Delete emoji
    # message[0] = re.sub(r'(\uD83C[\uDDE6-\uDDFF]\uD83C[\uDDE6-\uDDFF])', '', message[0])
    # Delete usernames and hastags hastags
    message[0] = re.sub(r'(@[A-Za-z0-9_.]+)|(#[A-Za-z0-9_]+)', '', message[0])
    # Every line will be tokenized into a word without taking into account the punctuation marks
    message[0] = message[0].translate(str.maketrans(dict.fromkeys(punctuation_marks, ' ')))
    
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(message[0]))
    wordnet_tagged = map(lambda x: (x[0], nltk_pos_tagger(x[1])), nltk_tagged)
    for word, tag in wordnet_tagged:
      if (word not in en_stops):
        if (word.isalpha()):
          word = word.lower()
          if tag is None:
            # data.append(word)
            parsedMessage += word + ' '
          else:        
            # data.append(lemmatizer.lemmatize(word, tag))
            parsedMessage += lemmatizer.lemmatize(word, tag) + ' '
    
    data.append(parsedMessage)
  return data







def processModel(fileLines):
  fileLines.pop(0)
  fileLines.pop(0)
  data = {}

  for line in fileLines:
    wordData = {'text': '', 'freq': 0, 'logProb': 0}
    splitted = line.split(' ')
    wordData['text'] = splitted[1]

    wordData['freq'] = int(splitted[3])
    wordData['logProb'] = float(splitted[-1])
    data[splitted[1]] = wordData
  return data


def  classify(messages, positiveModel, negativeModel):
  result = list()
  for message in messages:
    result.append({
      'text' : message[0:10],
      'posProb': 0,
      'negProb': 0,
      'class':  None
    })
    posProb = 1
    negProb = 1
    splitMessage = message.split(' ')
    for word in splitMessage:
      if word in positiveModel and word in negativeModel:
        posProb += positiveModel[word]['logProb']
        negProb += negativeModel[word]['logProb']
      else:
        posProb += positiveModel['__unknown__']['logProb']
        negProb += negativeModel['__unknown__']['logProb']
    result[-1]['posProb'] = posProb
    result[-1]['negProb'] = negProb
    if posProb > negProb:
      result[-1]['class'] = 'P'
    else:
      result[-1]['class'] = 'N'
  return result



def checkError():
  print("\nLoading data to classify...\n")
  # df = pd.read_excel(r"COV_test_g2_debug.xlsx",  sheet_name=0, dtype=str)
  df = pd.read_excel(r"COV_test_g2_debug.xlsx", header=None, dtype=str)
  # rawMessages = df[['Message']].to_numpy(dtype='str')
  # expected = df[['Emotion']].to_numpy(dtype='str')
  rawMessages = df[[1]].to_numpy(dtype='str')
  expected = df[[2]].to_numpy(dtype='str')

  print('\nPreprocessing messages to clasify...\n')
  messages = read_and_tokenize(rawMessages)

  print("\nLoading models...\n")
  positiveRaw = []
  negativeRaw = []
  with open('modelo_lenguaje_P.txt') as f:
    positiveRaw = f.read().splitlines()
  with open('modelo_lenguaje_N.txt') as f:
    negativeRaw = f.read().splitlines()

  positiveModel = processModel(positiveRaw)
  negativeModel = processModel(negativeRaw)
  
  print('\nClassifying...\n')
  classification = classify(messages, positiveModel, negativeModel)
  hits = 0
  for i in range(len(classification)):
    if classification[i]['class'] == expected[i][0][0:1]:
      hits += 1
  classificationFilename = 'clasificacion_alu0101318318.txt'
  
  resumenFilename = 'resumen_alu0101318318.txt'

  with open(classificationFilename, 'w') as f:
    for item in classification:
      formatPosProb = "{:.2f}".format(item['posProb'])
      formatNegProb = "{:.2f}".format(item['negProb'])
      f.write(f"{item['text']}, {formatPosProb}, {formatNegProb}, {item['class']}\n")

  with open(resumenFilename, 'w') as f:
    for item in classification:
      f.write(f"{item['class']}\n")
  print('Error: ', hits / len(classification) * 100)




def main():
  mode = 1
  if mode == 1:
    print("\nLoading data to classify...\n")
    df = pd.read_excel(r"COV_test_g2.xlsx", header=None, dtype=str)
    rawMessages = df.to_numpy(dtype='str')

    print('\nPreprocessing messages to clasify...\n')
    messages = read_and_tokenize(rawMessages)

    
    print("\nLoading models...\n")

    positiveRaw = []
    negativeRaw = []

    with open('modelo_lenguaje_P.txt') as f:
      positiveRaw = f.read().splitlines()

    with open('modelo_lenguaje_N.txt') as f:
      negativeRaw = f.read().splitlines()

    positiveModel = processModel(positiveRaw)
    negativeModel = processModel(negativeRaw)

    print('\nClassifying...\n')

    classification = classify(messages, positiveModel, negativeModel)

    classificationFilename = 'clasificacion_alu0101318318.txt'
    resumenFilename = 'resumen_alu0101318318.txt'

    with open(classificationFilename, 'w') as f:
      for item in classification:
        formatPosProb = "{:.2f}".format(item['posProb'])
        formatNegProb = "{:.2f}".format(item['negProb'])
        f.write(f"{item['text']}, {formatPosProb}, {formatNegProb}, {item['class']}\n")

    with open(resumenFilename, 'w') as f:
      for item in classification:
        f.write(f"{item['class']}\n")
  else:
    checkError()


main()
