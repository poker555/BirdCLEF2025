import os
import pandas as pd
import json

train_csv_path = 'C:/Users/brad9/Desktop/BirdCLEF/train.csv'
taxonomy_csv_path = 'C:/Users/brad9/Desktop/BirdCLEF/taxonomy.csv'

merge = pd.merge(
    pd.read_csv(train_csv_path),
    pd.read_csv(taxonomy_csv_path),
    on='primary_label',
    how='left'
)

columns_to_drop = ['type','url','latitude','longitude','license','author','secondary_labels']
merge = merge.drop(columns=columns_to_drop)

merge.to_csv('C:/Users/brad9/Desktop/BirdCLEF/merged_train.csv', index=False)
