#!/usr/bin/python
# -*- coding: utf-8 -*-
# Proyecto sobre Procesamiento de Lenguaje Natural para Inteligencia Artificial Avanzada
# Autor: Pablo Perez Gonzalez (alu0101318318@ull.edu.es)


import math
import pandas as pd
import re
import string

from copy import deepcopy



def modelProcess(messages, vocabulary):
  result = {
    'tweetsNumber': len(messages),
    'wordsNumber': 0,
    'corpus': {}
  }

  for word in vocabulary:
    wordData = {'text': word, 'freq': 0, 'logProb': 0}
    result['corpus'][word] = wordData

  result['corpus']['__unknown__'] = {'text': '__unknown__', 'freq': 0, 'logProb': 0}
  wordsInMessages = 0

  for message in messages:
    message[0] = re.sub(r'http\S+', '', message[0])
    message[0] = re.sub(r'<[^>]+>', '', message[0])
    message[0] = re.sub(r'(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)', '', message[0])
    for word in message[0].split(' '):
      if word in result['corpus']:
        result['corpus'][word]['freq'] += 1
      else:
        result['corpus']['__unknown__']['freq'] += 1
      wordsInMessages += 1
  
  print (wordsInMessages)
  result['wordsNumber'] = wordsInMessages

  for key in result['corpus']:
    # Suavizado Laplaciano
    result['corpus'][key]['freq'] += 1
    # Formula: P(palabra) = n_veces(palabra) + 1 / todas_palabras_corpus + n_palabras_vocabulario
    # En mi caso el +1 lo hago antes por lo que no tengo que sumar el 1 ni n_palabras_vocabulario 
    result['corpus'][key]['logProb'] = math.log(result['corpus'][key]['freq'] / wordsInMessages)

  return result




def main():
  print("\nLoading...\n")
  df = pd.read_excel(r"COV_train.xlsx", sheet_name=0, dtype=str)
  mask = (df['Emotion'] == "Negative")
  negativeDf = df[mask]
  positiveDf = df[~mask]

  positiveMessages = positiveDf[['Message']].to_numpy(dtype='str')
  negativeMessages = negativeDf[['Message']].to_numpy(dtype='str')

  vocabulary = []
  with open('vocabulario.txt') as f:
    vocabulary = f.read().splitlines()

  # with open('vocabulario.txt') as f:
  #   vocabulary = f.readlines()
  
  vocabulary.pop(0)

  clonePos = deepcopy(vocabulary)
  cloneNeg = deepcopy(vocabulary)

  print("\nProcessing positive messages...\n")

  positiveData = modelProcess(positiveMessages, clonePos)

  print("\nProcessing negative messages...\n")

  negativeData = modelProcess(negativeMessages, cloneNeg)

  positiveFilename = 'modelo_lenguaje_P.txt'
  negativeFilename = 'modelo_lenguaje_N.txt'

  with open(positiveFilename, 'w') as f:
    f.write('Numero de documentos (tweets) del corpus :%s\n' % positiveData['tweetsNumber'])
    f.write('Número de palabras del corpus:%s\n' % positiveData['wordsNumber'])
    for item in positiveData['corpus']:
      # print(positiveData['corpus'][item])
      f.write(f"Palabra: {positiveData['corpus'][item]['text']} Freq: {positiveData['corpus'][item]['freq']} LogProb: {positiveData['corpus'][item]['logProb']}\n")

  with open(negativeFilename, 'w') as f:
    f.write('Numero de documentos (tweets) del corpus :%s\n' % negativeData['tweetsNumber'])
    f.write('Número de palabras del corpus:%s\n' % negativeData['wordsNumber'])
    for item in negativeData['corpus']:
      f.write(f"Palabra: {negativeData['corpus'][item]['text']} Freq: {negativeData['corpus'][item]['freq']} LogProb: {negativeData['corpus'][item]['logProb']}\n")


main()

