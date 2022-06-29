import numpy as np
import pandas as pd
import pickle5 as pickle 
import os
import warnings
import sys
from sklearn.utils import shuffle

dataset_mal = './../feature_vector/input_mal.csv'
dataset_ben = './../feature_vector/input_ben.csv'

#insert class
malware = pd.read_csv(dataset_mal)
print(malware.shape)
benign = pd.read_csv(dataset_ben)
if not 'Class' in malware.columns:
    malware['Class'] = 1
    malware.to_csv(dataset_mal, index = False)
if not 'Class' in benign.columns:
    benign['Class'] = 0
    benign.to_csv(dataset_ben, index = False)

#shuffle
malware = shuffle(malware)
benign = shuffle(benign)

#split 4:4:2
k = 3
size_mal = int(malware.shape[0] * 40 / 100)
size_ben = int(benign.shape[0] * 40 / 100)
 
for i in range(k):
	try:
		globals()[f'malware{i}'] = malware[size_mal*i:size_mal*(i+1)]
		globals()[f'benign{i}'] = benign[size_ben*i:size_ben*(i+1)]
	except:
		globals()[f'malware{i}'] = malware[size_mal*(k-1):]
		globals()[f'benign{i}'] = benign[size_ben*(k-1):]
	result = pd.concat([globals()[f'malware{i}'], globals()[f'benign{i}']], ignore_index=True, axis=0).fillna(0)
	result = shuffle(result)
	'''for column in result.columns[1:]:
		sumCol = result[column].sum(axis = 0, skipna = True)
		if sumCol <= 50:
			with open(file + '.txt', 'a') as writeFile:
				writeFile.write(f'{column} \n')
			result = result.drop(column, axis = 1)'''
	result.to_csv(f'./../feature_vector/dataset{i+1}.csv', index=False)

globals()[f'benign{i-1}'].to_csv(f'./../feature_vector/dataset{i}_train_ben.csv', index = False)
globals()[f'malware{i-1}'].to_csv(f'./../feature_vector/dataset{i}_train.csv', index = False)
globals()[f'malware{i}'].to_csv(f'./../feature_vector/dataset{i+1}_train.csv', index = False)
globals()[f'benign{i}'].to_csv(f'./../feature_vector/dataset{i+1}_test_ben.csv', index = False)

