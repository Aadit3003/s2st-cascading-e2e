""" This is one of the main modules and contains the code to forward-feed 
the End-to-end S2ST model.

These functions are used to generate the translation files for the dev set
while comparing to the cascaded system. They are also reused in the 
live_s2st_demontration module.
"""
import time
import string
import time
import pandas as pd

import soundfile as sf
import torch
import librosa
from espnet_model_zoo.downloader import ModelDownloader
from espnet_recipe_scripts.s2st_inference import Speech2Speech
from espnet2.bin.asr_inference import Speech2Text as asr
from sacrebleu.metrics import BLEU

DATA_DIRECTORY = "/data/shire/data/aaditd/"

SOURCE_PATH = f"{DATA_DIRECTORY}speech_data/source_dataset/clips_petite/"
PREDICTION_PATH = f"{DATA_DIRECTORY}speech_data/pred_oob_e2e/"

DEV_TARGET_DATASET_PATH = "./dev_dataset/dev_target.tsv" # Gold Reference
DEV_SOURCE_DATASET_PATH = "./dev_dataset/dev_source.tsv"

VOCODER_PATH = "./cvss-c_en_wavegan_hubert_vocoder/checkpoint-450000steps.pkl"


def text_normalizer(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

def forward_e2e_model(speech2speech, asr_model, current_filename):

   global PREDICTION_PATH
   global SOURCE_PATH
   start_time = time.time()
   a = 0
   fs = 16000
   speech, rate = sf.read(f"{SOURCE_PATH}{current_filename}")
   speech = librosa.resample(speech, rate, fs)
   tensor_speech = torch.tensor(speech, dtype=torch.double).unsqueeze(0).float()


   length = tensor_speech.new_full([1], dtype=torch.long, fill_value=tensor_speech.size(1))
   output_dict = speech2speech(tensor_speech, length)


   output_wav = output_dict["wav"].cpu().numpy()
   extra = "_e2e_oob.wav"
   title = current_filename + extra
   sf.write(
      
       f"{PREDICTION_PATH}{title}",
       output_wav,
       fs,
       "PCM_16",
   )
#    print(f"Saved output for E2E OOB Model!! Took {round(time.time() - start_time, 2)} seconds!")

   pred_text, *_ = asr_model(output_wav)[0]


   return pred_text

def main():
    
    # ASR MODEL
    tag = "asapp/e_branchformer_librispeech"
    d = ModelDownloader()
    
    asr_instance = asr(
        **d.download_and_unpack(tag),
        device="cuda",
        minlenratio=0.0,
        maxlenratio=0.0,
        ctc_weight=0.3,
        beam_size=10,
        batch_size=0,
        nbest=1
    )

    bleu_metric = BLEU(effective_order=True)

    gold_df = pd.read_csv(DEV_TARGET_DATASET_PATH, header=None)

    Gold_Dict = {}
    count = 0
    for _, row in gold_df.iterrows():
        wav_file, ref_text = row[0].split("\t")
        Gold_Dict[wav_file] = ref_text
    
    print(f"Gold Dataset contains {len(Gold_Dict)} items")

    # Speech2Speech (S2ST) Translation Model
    
    lang = "es"
    fs = 16000
    model_info = d.download_and_unpack("espnet/jiyang_tang_cvss-c_es-en_discrete_unit")

    speech2speech = Speech2Speech(
        model_file=model_info["s2st_model_file"],
        train_config=model_info["s2st_train_config"],
        minlenratio=0.0,
        maxlenratio=4,
        beam_size=3,
        vocoder_file=VOCODER_PATH,
        device="cuda",
    )

    # Iterate through the Dev dataset files
    source_df = pd.read_csv(DEV_SOURCE_DATASET_PATH, sep = '\t')

    output_csv = {"Prediction": [], "Gold":[], "ASR_BLEU":[]}

    count = 1000
    for filename in list(source_df["path"]): # Alternatively, use os.listdir(SOURCE_PATH)

        ref_text = Gold_Dict[filename]
        
        text = forward_e2e_model(speech2speech = speech2speech, 
                                 asr_model = asr_instance, 
                                 current_filename = filename)
        
        score = bleu_metric.sentence_score(text_normalizer(text), [text_normalizer(ref_text)])

        output_csv["Prediction"].append(text_normalizer(text))
        output_csv["Gold"].append(text_normalizer(ref_text))
        output_csv['ASR_BLEU'].append(score)
        output_csv['File'].append(filename)


        count += 1
        if count%100 == 0: # Save every 100 iterations (optional)
            print(f"{i} files done!!")
            df2 = pd.DataFrame(output_csv)
            df2.to_csv(f".results/e2e_results/e2e_output_{i}.csv")


    results_df = pd.DataFrame(output_csv)
    results_df.to_csv("./results/e2e_results/e2e_output_final.csv")

    print("DONE!!")
    
if __name__ == "__main__":
    
    if not torch.cuda.is_available():
        raise RuntimeError("Please use GPU for better inference speed.")
    
    main()