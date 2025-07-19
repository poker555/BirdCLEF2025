import os
from glob import glob
import pandas as pd
from tqdm import tqdm
import torch
from pyannote.audio import Pipeline
import soundfile as sf

AUDIO_DIR = r"C:\Users\brad9\Desktop\BirdCLEF++\train_audio"
OUT_CSV = 'human_voice_stat.csv'
HF_TOKEN = "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"

pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=HF_TOKEN)
if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))
else:
    pipeline.to(torch.device("cpu"))

def get_audio_duration(file_path):
    with sf.SoundFile(file_path) as f:
        duration = len(f) / f.samplerate
    return duration

def process_file(file_path):
    try:
        speech_segments = pipeline(file_path)
        total_speech_sec = 0
        for segment in speech_segments.itertracks(yield_label=True):
            start = segment[0].start
            end = segment[0].end
            total_speech_sec += (end - start)
        duration = get_audio_duration(file_path)
        return {
            'filename': os.path.relpath(file_path, AUDIO_DIR),
            'speech_sec': total_speech_sec,
            'duration': duration,
            'speech_ratio': total_speech_sec/duration if duration > 0 else 0
        }
    except Exception as e:
        print(f"處理失敗：{file_path}，錯誤：{e}")
        return {
            'filename': os.path.relpath(file_path, AUDIO_DIR),
            'speech_sec': -1,
            'duration': -1,
            'speech_ratio': -1
        }

def main():
    files = sorted(glob(os.path.join(AUDIO_DIR, "**", "*.ogg"), recursive=True))
    results = []
    for f in tqdm(files, desc="檢查中"):
        res = process_file(f)
        results.append(res)
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"分析完成，已存入 {OUT_CSV}")

if __name__ == "__main__":
    main()
