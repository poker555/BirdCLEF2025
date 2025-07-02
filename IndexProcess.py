import pandas as pd
import os
import json

train = pd.read_csv('C:/Users/brad9/Desktop/BirdCLEF/merged_train.csv')
train['class_label'] = pd.factorize(train['class_name'])[0]
train['bio_label'] = pd.factorize(train['primary_label'])[0]

train.to_csv('C:/Users/brad9/Desktop/BirdCLEF/merge_train.csv', index=False)
