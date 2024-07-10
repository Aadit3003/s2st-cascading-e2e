""" This is one of the main modules and contains the code to forward-feed 
the End-to-end S2ST model.

These functions are used to generate the translation files for the dev set
while comparing to the cascaded system. They are also reused in the 
live_s2st_demontration module.
"""
import time
import torch
import string

import librosa
from espnet_model_zoo.downloader import ModelDownloader
from espnet_recipe_scripts.s2st_inference import Speech2Speech

# temporary for a buggy checkpoint
# cp /content/exp/s2st_stats_raw_es_en/train/src_feats_stats.npz /content/exp/s2st_stats_raw_es_en/train/tgt_feats_stats.npz 

# BLOCK 1
lang = "es"
fs = 16000

d = ModelDownloader()
model_info = d.download_and_unpack("espnet/jiyang_tang_cvss-c_es-en_discrete_unit")

# initiaite the speech2speech module
speech2speech = Speech2Speech(
    model_file=model_info["s2st_model_file"],
    train_config=model_info["s2st_train_config"],
    minlenratio=0.0,
    maxlenratio=4,
    beam_size=3,
    vocoder_file="/home/aaditd/2_Speech_Project/cvss-c_en_wavegan_hubert_vocoder/checkpoint-450000steps.pkl",
    device="cuda",
)



# speech2speech = Speech2Speech(
#     model_file="/data/shire/data/aaditd/trial/S2ST_Models/exp/s2st_train_s2st_discrete_unit_raw_fbank_es_en/362epoch.pth",
#     train_config="/data/shire/data/aaditd/trial/S2ST_Models/exp/s2st_train_s2st_discrete_unit_raw_fbank_es_en/config.yaml",
#     minlenratio=0.0,
#     maxlenratio=4,
#     beam_size=3,
#     vocoder_file="/data/shire/data/aaditd/trial/S2ST_Models/exp/unit_pretrained_vocoder/checkpoint-50000steps.pkl",
#     device="cuda",
# )

def text_normalizer(text):
    text = text.upper()
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

bleu = BLEU(effective_order=True)

print("S2ST MODEL:")

SOURCE_PATH = "/data/shire/data/aaditd/speech_data/source_dataset/clips/"
PRED_PATH = "/data/shire/data/aaditd/speech_data/pred_oob_e2e/"
GOLD_PATH = "/data/shire/data/aaditd/speech_data/target_dataset/dev.tsv"

ASR_SCORES = []

def text_normalizer(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))


df = pd.read_csv(GOLD_PATH, header=None)

Gold_Dict = {}
count = 0
for _, row in df.iterrows():
  wav_file, ref_text = row[0].split("\t")
  Gold_Dict[wav_file] = ref_text
  
print(len(Gold_Dict))

output_csv = {"Prediction": [], "Gold":[], "ASR_BLEU":[]}

# egs = pd.read_csv("ESPnet_st_egs/s2st/egs.csv")
i = 1000
for filename, ref_text in Gold_Dict.items():
    # print(filename)
    speech, rate = sf.read(SOURCE_PATH + filename)
    speech = librosa.resample(speech, rate, fs) # IMPORTANTTTT!!!! DON'T FORGET
    # assert rate == 16000, f"Actual speech rate is {rate} != 16000"
    tensor_speech = torch.tensor(speech, dtype=torch.double).unsqueeze(0).float()
    length = tensor_speech.new_full([1], dtype=torch.long, fill_value=tensor_speech.size(1))
    output_dict = speech2speech(tensor_speech, length)

    output_wav = output_dict["wav"].cpu().numpy()
    sf.write(
        f"{PRED_PATH}{filename}.wav",
        output_wav,
        fs,
        "PCM_16",
    )


    text, *_ = asr_model(output_wav)[0]
    # print(text_normalizer(text))
    # print(f"ASR hypothesis: {text_normalizer(text)}")
    # gold_text = Gold_Dict[filename.split(".wav")[0]]
    # print(text_normalizer(ref_text))
    
    score = bleu.sentence_score(text_normalizer(text), [text_normalizer(ref_text)])
    # print(score)
    # print()
    # print("*" * 50)



    output_csv["Prediction"].append(text_normalizer(text))
    output_csv["Gold"].append(text_normalizer(ref_text))
    output_csv['ASR_BLEU'].append(score)


    i += 1
    if i%100 == 0:
       print(f"{i} files done!!")
       df2 = pd.DataFrame(output_csv)
       df2.to_csv(f"/home/aaditd/2_Speech_Project/results/e2e_results/e2e_output_{i}.csv")

    if i > 2000:
       break


print("DONE!!")


df2 = pd.DataFrame(output_csv)
df2.to_csv("/home/aaditd/2_Speech_Project/results/e2e_results/e2e_output_final.csv")
# print(df2.head())

print("CSV WRITTEN!!")
