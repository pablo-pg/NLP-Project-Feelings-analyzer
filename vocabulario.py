#!/usr/bin/python
# -*- coding: utf-8 -*-
# Proyecto sobre Procesamiento de Lenguaje Natural para Inteligencia Artificial Avanzada
# Autor: Pablo Perez Gonzalez (alu0101318318@ull.edu.es)

# import contextualSpellCheck
import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
import string
# import spacy

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()
# nlp = spacy.load("en_core_web_sm")
# contextualSpellCheck.add_to_pipe(nlp)

def read_and_tokenize(rawMessages):
  punctuation_marks = string.punctuation + '…' + '”' + '“' + '-' + '‘'+ '’' + '´' + '—' + '`'
  data = []
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
    data += word_tokenize(message[0].translate(str.maketrans(dict.fromkeys(punctuation_marks, ' '))))
  return data

def preprocessing_words(data):
  en_stops = set(stopwords.words('english'))
  output_list = set()
  for word in data:
    # Only alphabetics strings will be processed
    if (word.isalpha()):
      word = word.lower()
      # Stopwords will be ignored
      if (word not in en_stops):
        # word = lemmatizer.lemmatize(word)
        output_list.add(lemmatizer.lemmatize(word))
  output_list = sorted(output_list)
  return output_list


def main():
  print("\nLoading...\n")
  df = pd.read_excel(r"COV_train.xlsx", sheet_name=0, dtype=str)
  rawMessages = df[['Message']].to_numpy(dtype='str')
  output_file = 'vocabulario.txt'
  data_list = read_and_tokenize(rawMessages)
  final_list = preprocessing_words(data_list)
  with open(output_file, 'w') as f:
    f.write('Numero de palabras: %s\n' % len(final_list))
    for item in final_list:
      f.write("%s\n" % item)


main()

