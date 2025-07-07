import pandas as pd

df = pd.read_csv("C:/Users/brad9/Desktop/BirdCLEF/merge_train.csv")

count = df['primary_label'].unique()
aves = df[df['class_name'] == 'Aves']
aves_count = aves['primary_label'].unique()

amphibia = df[df['class_name'] == 'Amphibia']
amphibia_count = amphibia['primary_label'].unique()

mammalia = df[df['class_name'] == 'Mammalia']
mammalia_count = mammalia['primary_label'].unique()
insecta = df[df['class_name'] == 'Insecta']
insecta_count = insecta['primary_label'].unique()


print("Number of unique primary labels:", len(count),len(aves_count),len(amphibia_count),len(mammalia_count),len(insecta_count))
