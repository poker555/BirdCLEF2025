import pandas as pd
import os
import json

def generate_dual_label_mappings_and_masks(taxonomy_csv_path: str, output_dir: str):
    # 絕對路徑處理
    taxonomy_csv_path = os.path.abspath(taxonomy_csv_path)
    output_dir = os.path.abspath(output_dir)
    mask_dir = os.path.join(output_dir, "organism_masks")
    os.makedirs(mask_dir, exist_ok=True)

    # 讀取 taxonomy.csv
    taxonomy_df = pd.read_csv(taxonomy_csv_path)

    # === 物種分類：class_name ===
    species_classes = sorted(taxonomy_df["class_name"].unique())
    species_class_to_idx = {cls: idx for idx, cls in enumerate(species_classes)}
    idx_to_species_class = {idx: cls for cls, idx in species_class_to_idx.items()}

    # === 生物分類：primary_label ===
    organism_classes = sorted(taxonomy_df["primary_label"].unique())
    organism_class_to_idx = {cls: idx for idx, cls in enumerate(organism_classes)}
    idx_to_organism_class = {idx: cls for cls, idx in organism_class_to_idx.items()}

    # 儲存標籤映射
    with open(os.path.join(output_dir, "species_class_to_idx.json"), "w") as f:
        json.dump(species_class_to_idx, f)
    with open(os.path.join(output_dir, "idx_to_species_class.json"), "w") as f:
        json.dump(idx_to_species_class, f)

    with open(os.path.join(output_dir, "organism_class_to_idx.json"), "w") as f:
        json.dump(organism_class_to_idx, f)
    with open(os.path.join(output_dir, "idx_to_organism_class.json"), "w") as f:
        json.dump(idx_to_organism_class, f)

    # 為每個物種建立 organism 遮罩
    for species in species_classes:
        sub_df = taxonomy_df[taxonomy_df["class_name"] == species]
        organism_labels = sub_df["primary_label"].unique()
        indices = [organism_class_to_idx[label] for label in organism_labels if label in organism_class_to_idx]

        mask_path = os.path.join(mask_dir, f"class_mask_species_{species}.json")
        with open(mask_path, "w") as f:
            json.dump(indices, f)

    print(f"✅ 映射與遮罩已完成！共 {len(species_classes)} 個物種分類、{len(organism_classes)} 個生物分類")
    print(f"📁 輸出資料夾：{output_dir}")
    print(f"📄 使用 taxonomy 檔案：{taxonomy_csv_path}")

# ========= 範例執行 =========
if __name__ == "__main__":
    generate_dual_label_mappings_and_masks(
        taxonomy_csv_path="C:/Users/brad9/Desktop/BirdCLEF/taxonomy.csv",
        output_dir="C:/Users/brad9/Desktop/BirdCLEF/index"
    )
