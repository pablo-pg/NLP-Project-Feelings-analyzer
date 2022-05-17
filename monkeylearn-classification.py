#!/usr/bin/python
# -*- coding: utf-8 -*-
# Proyecto sobre Procesamiento de Lenguaje Natural para Inteligencia Artificial Avanzada
# Autor: Pablo Perez Gonzalez (alu0101318318@ull.edu.es)

import pandas as pd
import numpy as np
from monkeylearn import MonkeyLearn


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



ml = MonkeyLearn('f280dedd11b4b5c26f6ce78898e69f4cb75650a9')

df = pd.read_excel(r"COV_train.xlsx",  sheet_name=0, dtype=str)
rawMessages = df[['Message']].to_numpy(dtype='str')
# print (rawMessages[0])
data = normalize(rawMessages)
# print (data)
model_id = 'cl_avq7AcJT'
result = ml.classifiers.classify(model_id, data)
print(result.body)