""" This module contains the code for the live demonstration of the 
S2ST (Speech-to-Speech Translation) system, for a single soundfile (either the path may be specified
or it may be recorded live).

It uses the forward_feed functions for the E2E and Cascaded models (both out-of-box and fine-tuned),
on the soundfile and returns the translated .wav files, as well as a csv file with
the five Speech Translation metrics (ASR-BLEU, COMET, METEOR, BLASER-2.0) specified in the expanded_translation_metrics module.
"""
import numpy as np
import pandas as pd
import librosa
import time
import glob
import os
import string
from string import punctuation

import gradio as gr
import kaldiio
import soundfile as sf
import torch
import espnetez as ez
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.s2t_inference import Speech2Text
from espnet2.bin.asr_inference import Speech2Text as asr
from espnet2.bin.s2t_inference_language import Speech2Language
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
from s2st_inference import Speech2Speech
from espnet2.layers.create_adapter_fn import create_lora_adapter
import evaluate
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.models.blaser.loader import load_blaser_model
from sacrebleu.metrics import BLEU

from expanded_translation_metrics import comet_score_vectorized, blaser_score_vectorized, meteor_score_vectorized
from utils.macro_average_results import segment_bleu_score_string
from forward_feed_cascaded_finetuned_oob import forwadrd_cascaded_model, text_normalizer
from forward_feed_e2e import forward_e2e_model

# CONSTANTS

DATA_DIRECTORY = "/data/shire/data/aaditd/"
  
SOURCE_PATH = f"{DATA_DIRECTORY}speech_data/source_dataset/clips/"
PREDICTION_PATH = "./demonstration/output/"

demo_sample_filename = "common_voice_es_19749502.mp3"

SOURCE_TEXT = "Estas instalaciones se encuentran incluidas dentro de un parque recreativo y deportivo"
REF_TEXT = "this facilities are included inside a recreational and sports park"

# MODELS
FINETUNED_MODEL_PATH = f"{DATA_DIRECTORY}speech_data/exp-finetuned/2epoch.pth"
VOCODER_PATH = "./cvss-c_en_wavegan_hubert_vocoder/checkpoint-450000steps.pkl"
TTS_MODEL_PATH = ".tts_multi-speaker_model/exp/tts_train_full_band_multi_spk_vits_raw_phn_tacotron_g2p_en_no_space/train.total_count.ave_10best.pth"
TTS_TRAIN_CONFIG_PATH = "./tts_config.yaml"


def score_texts_with_all_metrics(source_texts, ref_texts, pred_texts, bleu, blaser, text_embedder, meteor, comet):
   
   list_asr_bleu = [segment_bleu_score_string(str(bleu.sentence_score(text_normalizer(p), [text_normalizer(r)]) )  ) for p, r in zip(pred_texts, ref_texts)]
   list_comet = comet_score_vectorized(pred_texts, ref_texts, source_texts, comet)
   list_meteor = [meteor_score_vectorized(p, r, meteor) for p, r in zip(pred_texts, ref_texts)]
   list_blaser = blaser_score_vectorized(source_texts=source_texts, pred_texts=pred_texts, blaser=blaser, text_embedder=text_embedder)

   return list_asr_bleu, list_comet, list_meteor, list_blaser


def main():
   # Metrics Models!
   bleu_metric = BLEU(effective_order=True)
   blaser_metric = load_blaser_model("blaser_2_0_qe").eval()
   text_embedder = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder")
   meteor_metric = evaluate.load('meteor')
   comet_metric = evaluate.load('comet')
   
   d = ModelDownloader()
   asr_model = asr(
       **d.download_and_unpack("asapp/e_branchformer_librispeech"),
       device="cuda",
       minlenratio=0.0,
       maxlenratio=0.0,
       ctc_weight=0.3,
       beam_size=10,
       batch_size=0,
       nbest=1
   )
   print("Step 1 DONE: Loaded Evaluation Models!")

   # SPEECH MODELS
   # Speech2Speech (S2ST) Translation Model
   lang = "es"
   fs = 16000
   s2s_model_info = d.download_and_unpack("espnet/jiyang_tang_cvss-c_es-en_discrete_unit")
   speech2speech = Speech2Speech(
       model_file=s2s_model_info["s2st_model_file"],
       train_config=s2s_model_info["s2st_train_config"],
       minlenratio=0.0,
       maxlenratio=4,
       beam_size=3,
       vocoder_file=VOCODER_PATH,
       device="cuda",
   )

   # Text2Speech (TTS) Model
   text2speech = Text2Speech.from_pretrained(
       train_config = TTS_TRAIN_CONFIG_PATH,
       model_file=TTS_MODEL_PATH,
       vocoder_tag=str_or_none("none"),
       device="cuda",
       threshold=0.5,
       minlenratio=0.0,
       maxlenratio=10.0,
       use_att_constraint=False,
       backward_window=1,
       forward_window=3,
       speed_control_alpha=1.0,
       noise_scale=0.333,
       noise_scale_dur=0.333,
   )

   # Speech2Text (S2T) Translation Model
   s2l = Speech2Language.from_pretrained(
       model_tag="espnet/owsm_v3.1_ebf",
       device="cuda",
       nbest=1,
   )

   speech2text_model = Speech2Text.from_pretrained(
       model_tag="espnet/owsm_v3.1_ebf",
       device="cuda",
       beam_size=5,
       ctc_weight=0.0,
       maxlenratio=0.0,
       # below are default values which can be overwritten in __call__
       lang_sym="<eng>",
       task_sym="<asr>",
       predict_time=False
   )
   LORA_TARGET = ["w_1", "w_2", "merge_proj"]
   print("Step 2 DONE: Loaded Speech Models!")
  
   # INFERENCE
   pred_text_e2e = forward_e2e_model(speech2speech=speech2speech,
                                     asr_model=asr_model,
                                     filename=demo_sample_filename)

   pred_text_casc_oob = forwadrd_cascaded_model(speech2text_model=speech2text_model,
                                            speech2language=s2l,
                                            text2speech_model=text2speech,
                                            asr_model=asr_model,
                                            current_filename=demo_sample_filename,
                                            use_finetuned=False)
  
   pred_text_casc_fine = forwadrd_cascaded_model(speech2text_model=speech2text_model,
                                            speech2language=s2l,
                                            text2speech_model=text2speech,
                                            asr_model=asr_model,
                                            current_filename=demo_sample_filename,
                                            use_finetuned=True,
                                            LORA_TARGET=LORA_TARGET,
                                            finetuned_s2t_model_path=FINETUNED_MODEL_PATH)
  
   print("Step 3 DONE: Performed Forward feeding")
   
   prediction_texts = [pred_text_e2e, pred_text_casc_oob, pred_text_casc_fine]
   source_texts = [SOURCE_TEXT for _ in range(3)]
   reference_texts = [REF_TEXT for _ in range(3)]

   # EVALUATION
   list_asr_bleu, list_comet, list_meteor, list_blaser = score_texts_with_all_metrics(source_texts=source_texts,
                   pred_texts=prediction_texts,
                   ref_texts=reference_texts,
                   bleu=bleu_metric,
                   blaser=blaser_metric,
                   text_embedder=text_embedder,
                   meteor=meteor_metric,
                   comet=comet_metric)
  
   list_bleu = [a for a, _, _ in list_asr_bleu]
   list_bp = [b for _, b, _ in list_asr_bleu]
   list_ratio = [c for _, _, c in list_asr_bleu]
  
   # SAVE RESULTS
   df_final = pd.DataFrame({"Filename": [demo_sample_filename for _ in range(3)],
               "Source_Text": source_texts,
               "Pred_Text": prediction_texts,
               "Ref_Text": reference_texts,
               "ASR_BLEU": list_bleu,
               "BP": list_bp,
               "Ratio": list_ratio,
               "COMET": list_comet,
               "METEOR": list_meteor,
               "BLASER2": list_blaser
           })
  
   print("Step 4 DONE: Evaluation metrics calculated")
   df_final.to_csv(f"{PREDICTION_PATH}metrics.csv")
   print()
  
   print("ALL DONE!")
   
if __name__ == "__main__":
    
    if not torch.cuda.is_available():
        raise RuntimeError("Please use GPU for better inference speed.")

    start_time = time.time() 
    main()
    print()
    print(f"Total Time = {round(time.time() - start_time, 2)} seconds!")