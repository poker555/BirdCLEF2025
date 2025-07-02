import pandas as pd

# 讀取你的合併資料
df = pd.read_csv("C:/Users/brad9/Desktop/BirdCLEF/merge_train.csv")

# 對每一個物種（organism_class）單獨建立細分類標籤欄
for organism in df["class_label"].unique():
    mask = df["class_label"] == organism
    # 用 species_class 建立新的連續標籤
    species_list = df.loc[mask, "primary_label"].unique()
    mapping = {species: i for i, species in enumerate(species_list)}
    # 新增一個標籤欄位（名稱如 class_label_Aves）
    df.loc[mask, f"class_label_{organism}"] = df.loc[mask, "primary_label"].map(mapping)

# 儲存結果
df.to_csv("merge_train_with_class_labels.csv", index=False)
