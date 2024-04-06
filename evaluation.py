import time
import torch
import os
import string
from s2st_inference import Speech2Speech



def write_list_to_file(file, lines):
   with open(file, 'w') as f:
    for line in lines:
        f.write(f"{line}\n")


def text_normalizer(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))


# BLOCK 2
import time
import torch
import string
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text as asr

tag = "asapp/e_branchformer_librispeech"

d = ModelDownloader()
# It may takes a while to download and build models
asr_model = asr(
    **d.download_and_unpack(tag),
    device="cuda",
    minlenratio=0.0,
    maxlenratio=0.0,
    ctc_weight=0.3,
    beam_size=10,
    batch_size=0,
    nbest=1
)


# BLOCK 3

import torch
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from sacrebleu.metrics import BLEU

bleu = BLEU(effective_order=True)


first_n = 5
# PRED_PATH = "/data/shire/data/aaditd/speech_data/dummy"
PRED_PATH = "/home/aaditd/2_Speech_Project/pred_oob_casc_RAW"
GOLD_PATH = "/data/shire/data/aaditd/speech_data/target_dataset/dev.tsv"

# PRED_FILES = ["/data/shire/data/aaditd/speech_data/pred_oob_cascaded/common_voice_es_19128864.mp3_changed.wav"]

df = pd.read_csv(GOLD_PATH, header=None)

Gold_Dict = {}
for _, row in df.iterrows():
  wav_file, ref_text = row[0].split("\t")
  Gold_Dict[wav_file] = ref_text

PRED_FILES = os.listdir(PRED_PATH)
# assert len(df) == len(PRED_FILES), f"Different Number of files in df ({len(df)}) and pred_cascaded_oob ({len(PRED_FILES)})"

PRED_FILES = [p for p in PRED_FILES if "16000_mono_16bit" not in p]
NEW_PRED_FILES = os.listdir("/home/aaditd/2_Speech_Project/pred_oob_casc_RAW")
NEW_PRED_FILES = [p for p in NEW_PRED_FILES if "16000_mono_16bit" not in p] # WE want prefixes only
print(len(NEW_PRED_FILES))
# print(PRED_FILES)

ASR_SCORES = []
output_csv = {"Prediction": [], "Gold":[], "ASR_BLEU":[]}

i = 1000
for pred_file in NEW_PRED_FILES:
   
   prefix = pred_file.split(".wav")[0]
   
   name = f"{PRED_PATH}/{prefix}.wav"
   # print(f"Reading {name}")
   # speech, rate = sf.read(name)
   speech, rate = sf.read(f"{PRED_PATH}/{prefix}_16000_mono_16bit.wav")
   # speech, rate = sf.read(f"{pred_file}")
   # print(rate)
   assert rate == 16000, f"SR doesn't match {rate}"
   # print("READ SPEECH: ", speech)
   # print(type(speech))
   # pred_wav = torch.tensor(speech, dtype=torch.double)
   # print(pred_wav.shape)
   # print(f"Filename: {pred_file}")

   pred_text, *_ = asr_model(speech)[0]
   # print(f"Pred Text: {pred_text}")

   gold_text = Gold_Dict[pred_file.split(".wav")[0]]
   # gold_text = Gold_Dict["common_voice_es_19128864.mp3"]
   # print(f"Gold Text: {gold_text}")

   score = bleu.sentence_score(text_normalizer(pred_text), [text_normalizer(gold_text)])
   # print(f"Score: {score}")
   ASR_SCORES.append(score)

   output_csv["Prediction"].append(text_normalizer(pred_text))
   output_csv["Gold"].append(text_normalizer(gold_text))
   output_csv['ASR_BLEU'].append(score)
   i += 1
   if i%100 == 0:
      print(f"{i} files done!!")
      df2 = pd.DataFrame(output_csv)
      df2.to_csv(f"/home/aaditd/2_Speech_Project/results/casc_results/casc_output_{i}.csv")

print("DONE!!")

df2 = pd.DataFrame(output_csv)
df2.to_csv(f"/home/aaditd/2_Speech_Project/results/casc_results/casc_output_final.csv")


   