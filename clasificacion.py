#!/usr/bin/python
# -*- coding: utf-8 -*-
# Proyecto sobre Procesamiento de Lenguaje Natural para Inteligencia Artificial Avanzada
# Autor: Pablo Perez Gonzalez (alu0101318318@ull.edu.es)



def processModel(fileLines):
  fileLines.pop(0)
  fileLines.pop(0)
  data = []
  # print(fileLines[0])
  # print()
  # print()
  for line in fileLines:
    wordData = {'text': '', 'freq': 0, 'logProb': 0}
    splitted = line.split(' ')
    wordData['text'] = splitted[1]
    # wordData['freq'] = splitted[3]
    # wordData['logProb'] = splitted[-1]
    # print (splitted)

    wordData['freq'] = int(splitted[3])
    wordData['logProb'] = float(splitted[-1])
    data.append(wordData)
  return data






def main():

  positiveRaw = []
  negativeRaw = []

  with open('modelo_lenguaje_P.txt') as f:
    positiveRaw = f.read().splitlines()

  with open('modelo_lenguaje_N.txt') as f:
    negativeRaw = f.read().splitlines()

  positiveModel = processModel(positiveRaw)
  negativeModel = processModel(negativeRaw)

  print(len(positiveModel))
  print(len(negativeModel))


main()
