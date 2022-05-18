#!/usr/bin/python
# -*- coding: utf-8 -*-
# Proyecto sobre Procesamiento de Lenguaje Natural para Inteligencia Artificial Avanzada
# Autor: Pablo Perez Gonzalez (alu0101318318@ull.edu.es)

import pandas as pd
import numpy as np
# from monkeylearn import MonkeyLearn


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



# ml = MonkeyLearn('f280dedd11b4b5c26f6ce78898e69f4cb75650a9')

df = pd.read_excel(r"COV_train.xlsx",  sheet_name=0, dtype=str)
rawMessages = df[['Message']].to_numpy(dtype='str')
data = normalize(rawMessages)
target = normalize(df[['Emotion']].to_numpy(dtype='str'))
# MONKEY LEARN

# model_id = 'cl_avq7AcJT'
# result = ml.classifiers.classify(model_id, data)
# print(result.body)

# VADER

# import nltk
# nltk.download('vader_lexicon')
# from nltk.sentiment.vader import SentimentIntensityAnalyzer

# for sentence in data:
#   sid = SentimentIntensityAnalyzer()
#   ss = sid.polarity_scores(sentence)
#   for k in sorted(ss):
#       print('{0}: {1}, '.format(k, ss[k]), end='')
#   print()


# NLTK
import nltk
def punc_clean(text):
    import string as st
    a=[w for w in text if w not in st.punctuation]
    return ''.join(a)
  
data = list(map(punc_clean, data))
# print(data[0])

def remove_stopword(text):
    stopword=nltk.corpus.stopwords.words('english')
    stopword.remove('not')
    a=[w for w in nltk.word_tokenize(text) if w not in stopword]
    return ' '.join(a)
# data = data.apply(remove_stopword)
data = list(map(remove_stopword, data))
# print(data[0])
print('Eliminados signos de puntuacion y stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
vectr = TfidfVectorizer(ngram_range=(1,2),min_df=1)
vectr.fit(data)
vect_X = vectr.transform(data)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# clf=model.fit(vect_X,data['sentiment'])
clf = model.fit(vect_X, target)
# clf.score(vect_X,data['sentiment'])*100
clf.score(vect_X, target)*100

print('Modelo entrenado')

result1 = clf.predict(vectr.transform(['I love icecream']))
result2 = clf.predict(vectr.transform(['My son died yesterday']))
print(result1, result2)
