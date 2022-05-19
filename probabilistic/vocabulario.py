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
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

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


lemmatizer = WordNetLemmatizer()
# nlp = spacy.load("en_core_web_sm")
# contextualSpellCheck.add_to_pipe(nlp)

def read_and_tokenize(rawMessages):
  #{ Part-of-speech constants
  ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
  #}
  POS_LIST = [NOUN, VERB, ADJ, ADV]
  en_stops = set(stopwords.words('english'))
  punctuation_marks = string.punctuation + '…' + '”' + '“' + '-' + '‘'+ '’' + '´' + '—' + '`'
  data = set()
  for message in rawMessages :
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
            data.add(word)
          else:        
            data.add(lemmatizer.lemmatize(word, tag))
  return sorted(data)


def main():
  print("\nLoading...\n")
  df = pd.read_excel(r"../COV_train.xlsx", sheet_name=0, dtype=str)
  rawMessages = df[['Message']].to_numpy(dtype='str')
  output_file = 'vocabulario.txt'
  data_list = read_and_tokenize(rawMessages)
  # print(data_list)
  # final_list = preprocessing_words(data_list)
  with open(output_file, 'w') as f:
    f.write('Numero de palabras: %s\n' % len(data_list))
    for item in data_list:
      f.write("%s\n" % item)


main()

