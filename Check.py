from pathlib import Path
import numpy as np
from collections import Counter
from tqdm import tqdm

# === 設定資料夾路徑 ===
log_mel_dir = Path("C:/Users/brad9/Desktop/BirdCLEF/log_mel")  # ← 修改為你的資料夾路徑

shape_counter = Counter()
bad_files = []

# === 檢查所有 .npy 檔案（逐一 + 進度條） ===
npy_files = list(log_mel_dir.rglob("*.npy"))
print(f"🔍 開始檢查 {len(npy_files)} 個 .npy 檔案...\n")

for path in tqdm(npy_files, desc="檢查中"):
    try:
        mel = np.load(path)
        shape = mel.shape
        shape_counter[shape] += 1
        if mel.ndim != 2:
            print(f"⚠️ 非 2D: {shape} | {path}")
            bad_files.append((path, shape))
    except Exception as e:
        print(f"❌ 無法讀取: {path} | {e}")
        bad_files.append((path, f"錯誤: {e}"))

# === 統計結果 ===
print("\n=== ✅ 維度統計 ===")
for shape, count in shape_counter.items():
    print(f"Shape {shape}: {count} 檔案")

# === 異常摘要 ===
if bad_files:
    print(f"\n⚠️ 總共發現 {len(bad_files)} 個異常檔案（包含非 2D 或無法讀取）")
else:
    print("🎉 所有 .npy 檔案皆為 2D 且格式正確。")
