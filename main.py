# Proyecto sobre Procesamiento de Lenguaje Natural para Inteligencia Artificial Avanzada
# Autor: Pablo Pérez González (alu0101318318@ull.edu.es)

import contextualSpellCheck
from numpy import append
import pandas as pd
import re
import spacy
import string

nlp = spacy.load("en_core_web_sm")
contextualSpellCheck.add_to_pipe(nlp)

df = pd.read_excel(r"COV_train.xlsx", sheet_name=0, dtype=str)
# messages = pd.DataFrame(df, columns= ['Message'])

rawMessages = df[['Message']].to_numpy(dtype='str')

preProcessed = []
for message in rawMessages :
    # Se eliminan los links
    message[0] = re.sub(r'http\S+', '', message[0])
    # Se eliminan las etiquetas html
    message[0] = re.sub(r'<[^>]+>', '', message[0])
    # Se eliminan los emoji
    message[0] = re.sub(r'(\uD83C[\uDDE6-\uDDFF]\uD83C[\uDDE6-\uDDFF])', '', message[0])
    # Se eliminan los nombres de usuario y los hastags
    message[0] = re.sub(r'(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)', '', message[0])
# ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split())
    # Se eliminan los signos de puntuacion, se pasa a minusculas y se quitean los saltos de linea
    strNoPoints = message[0].translate(str.maketrans('', '', string.punctuation)).lower().replace('\n', ' ')
    preProcessed.append(strNoPoints)

print(preProcessed[0])

correct = []

for message in preProcessed:
  doc = nlp(preProcessed[0])
  print(doc._.performed_spellCheck)
  # correct.append(doc._.outcome_spellCheck)
  if doc._.performed_spellCheck == False:
    for token in doc._.outcome_spellCheck:
      correct.append(token.lemma_)
  else:
    for token in doc:
      correct.append(token.lemma_)
  #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
  #             token.shape_, token.is_alpha, token.is_stop)

print(correct, sep='\n')

with open('vocabulario.txt', 'w') as f:
  for item in correct:
    f.write("%s\n" % item)
