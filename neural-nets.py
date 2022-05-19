#!/usr/bin/python
# -*- coding: utf-8 -*-
# Proyecto sobre Procesamiento de Lenguaje Natural para Inteligencia Artificial Avanzada
# Autor: Pablo Perez Gonzalez (alu0101318318@ull.edu.es)

import pandas as pd
import numpy as np


def normalize(datas): 
  result = []
  for message in datas:
    if isinstance(message, np.ndarray):
      result.append(message[0])
    elif isinstance(message, str):
      result.append(message)
    else:
      print(type(message))
  return result


df = pd.read_excel(r"COV_train.xlsx",  sheet_name=0, dtype=str)
rawMessages = df[['Message']].to_numpy(dtype='str')
data = normalize(rawMessages)
target = normalize(df[['Emotion']].to_numpy(dtype='str'))

# MONKEY LEARN

# from monkeylearn import MonkeyLearn
# ml = MonkeyLearn('f280dedd11b4b5c26f6ce78898e69f4cb75650a9')
# model_id = 'cl_avq7AcJT'
# result = ml.classifiers.classify(model_id, data)
# print(result.body)


# NLTK

import nltk
def punc_clean(text):
    import string as st
    a=[w for w in text if w not in st.punctuation]
    return ''.join(a)
  
print('Eliminando signos de puntuación del modelo')
data = list(map(punc_clean, data))

def remove_stopword(text):
    stopword=nltk.corpus.stopwords.words('english')
    stopword.remove('not')
    a=[w for w in nltk.word_tokenize(text) if w not in stopword]
    return ' '.join(a)
print('Eliminando stopwords del modelo')
data = list(map(remove_stopword, data))


from sklearn.feature_extraction.text import TfidfVectorizer
vectr = TfidfVectorizer(ngram_range=(1,2),min_df=1)
print('Entrenando el modelo')
vectr.fit(data)
vect_X = vectr.transform(data)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
clf = model.fit(vect_X, target)
clf.score(vect_X, target)*100

print('Modelo entrenado')


# df = pd.read_excel(r"COV_test_g2_debug.xlsx", header=None, dtype=str)
# rawMessages = df[[1]].to_numpy(dtype='str')
# expected = df[[2]].to_numpy(dtype='str')
# target = normalize(expected)
df = pd.read_excel(r"COV_test_g2.xlsx", header=None, dtype=str)
rawMessages = df.to_numpy(dtype='str')

data = normalize(rawMessages)

print('Eliminando signos de puntuación del df a clasificar')
data = list(map(punc_clean, data))
print('Eliminando stopwords del df a clasificar')
data = list(map(remove_stopword, data))

def checkError():
  if len(data) == len(target):
    hits = 0
    for i in range(len(data)):
      prediction = clf.predict(vectr.transform([data[i]]))
      if (prediction == 'Positive' and target[i] == 'Positive') or (prediction == 'Negative' and target[i] == 'Negative'):
        hits += 1
    accuracy = "{:.2f}".format(hits / len(target) * 100)
    print(f'Accuracy: {accuracy}')

def classifyDf():
  result = []
  for i in range(len(data)):
    prediction = clf.predict(vectr.transform([data[i]]))
    classification = {"text": data[i][0:10], "class": ''}
    if prediction == 'Positive':
      classification["class"] = 'P'
    elif prediction == 'Negative':
      classification["class"] = 'N'
    result.append(classification)
  return result

if len(df.columns) >= 2:
  print('Calculando error')
  checkError()
else:
  print('Clasificando')
  classification = classifyDf()
  classificationFilename = 'clasificacion_2_alu0101318318.txt'
  resumenFilename = 'resumen_2_alu0101318318.txt'

  with open(classificationFilename, 'w') as f:
    for item in classification:
      f.write(f"{item['text']}, {item['class']}\n")

  with open(resumenFilename, 'w') as f:
    for item in classification:
      f.write(f"{item['class']}\n")
