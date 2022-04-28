#!/usr/bin/python
# -*- coding: utf-8 -*-
# Proyecto sobre Procesamiento de Lenguaje Natural para Inteligencia Artificial Avanzada
# Autor: Pablo Perez Gonzalez (alu0101318318@ull.edu.es)


import math
import pandas as pd
import re
import string



def modelProcess(messages, vocabulary):
  result = {
    'tweetsNumber': len(messages),
    'wordsNumber': len(vocabulary),
    'corpus': set()
  }

  for word in vocabulary:
    wordData = {'text': word, 'freq': 0, 'logProb': 0}
    result['corpus'].add({word: wordData})

  wordsInMessages = 0

  for word in messages:
    result['corpus'][word]['freq'] += 1
    wordsInMessages += 1
  
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
  mask = df['Message'] == "Negative"
  positiveDf = df[~mask]
  negativeDf = df[mask]

  positiveMessages = positiveDf[['Message']].to_numpy(dtype='str')
  negativeMessages = negativeDf[['Message']].to_numpy(dtype='str')

  vocabulary = []
  with open('vocabulario.txt') as f:
    vocabulary = f.readlines()
  vocabulary.pop(0)

  positiveData = modelProcess(positiveMessages, vocabulary)
  negativeData = modelProcess(negativeMessages, vocabulary)

  positiveFilename = 'modelo_lenguaje_P.txt'
  negativeFilename = 'modelo_lenguaje_N.txt'

  with open(positiveFilename, 'w') as f:
    f.write('Numero de documentos (tweets) del corpus :%s\n' % positiveData['tweetsNumber'])
    f.write('Número de palabras del corpus:%s\n' % positiveData['wordsNumber'])
    for item in positiveData['corpus']:
      f.write("Palabra: %s Freq: %s LogProb: %s\n" % item.text, item.freq, item.logProb)

  with open(negativeFilename, 'w') as f:
    f.write('Numero de documentos (tweets) del corpus :%s\n' % negativeData['tweetsNumber'])
    f.write('Número de palabras del corpus:%s\n' % negativeData['wordsNumber'])
    for item in negativeData['corpus']:
      f.write("Palabra: %s Freq: %s LogProb: %s\n" % item['text'], item['freq'], item['logProb'])


main()

