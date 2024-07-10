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

from expanded_translation_metrics import comet_score_vector, blaser_score_vector, meteor_score




# CONSTANTS
  
SOURCE_PATH = "/data/shire/data/aaditd/speech_data/source_dataset/clips/"
filename = "common_voice_es_19749502.mp3"


SOURCE_TEXT = "Estas instalaciones se encuentran incluidas dentro de un parque recreativo y deportivo"
REF_TEXT = "this facilities are included inside a recreational and sports park"


PREDICTION_PATH = "/home/aaditd/2_Speech_Project/demonstration/output/"
FINETUNED_MODEL_PATH = "/data/shire/data/aaditd/speech_data/exp-finetuned/2epoch.pth"


VOCODER_PATH = "/home/aaditd/2_Speech_Project/cvss-c_en_wavegan_hubert_vocoder/checkpoint-450000steps.pkl"
TTS_MODEL_PATH = "tts_multi-speaker_model/exp/tts_train_full_band_multi_spk_vits_raw_phn_tacotron_g2p_en_no_space/train.total_count.ave_10best.pth"
TTS_TRAIN_CONFIG_PATH = "/home/aaditd/2_Speech_Project/tts_config_old.yaml"




def text_normalizer(text):
   text = text.upper()
   return text.translate(str.maketrans('', '', string.punctuation))


if not torch.cuda.is_available():
   raise RuntimeError("Please use GPU for better inference speed.")


def get_stuff(string):
       parts = string.split(" ")
       main_score = float(parts[2])
       other_scores = [float(p) for p in parts[3].split("/")]
       bp = float(parts[6])
       ratio = float(parts[9])
       return (main_score, bp, ratio)


def get_all_metrics(source_texts, ref_texts, pred_texts, bleu, blaser, text_embedder, meteor, comet):
   list_asr_bleu = [get_stuff(str(bleu.sentence_score(text_normalizer(p), [text_normalizer(r)]) )  ) for p, r in zip(pred_texts, ref_texts)]
   list_comet = comet_score_vector(pred_texts, ref_texts, source_texts, comet)
   list_meteor = [meteor_score(p, r, meteor) for p, r in zip(pred_texts, ref_texts)]
   list_blaser = blaser_score_vector(source_texts=source_texts, pred_texts=pred_texts, blaser=blaser, text_embedder=text_embedder)




   return list_asr_bleu, list_comet, list_meteor, list_blaser




def forward_e2e_model(speech2speech, asr_model, filename):

   global PREDICTION_PATH
   global SOURCE_PATH
   start_time = time.time()
   a = 0
   fs = 16000
   speech, rate = sf.read(f"{SOURCE_PATH}{filename}")
   speech = librosa.resample(speech, rate, fs)
   tensor_speech = torch.tensor(speech, dtype=torch.double).unsqueeze(0).float()


   length = tensor_speech.new_full([1], dtype=torch.long, fill_value=tensor_speech.size(1))
   output_dict = speech2speech(tensor_speech, length)


   output_wav = output_dict["wav"].cpu().numpy()
   extra = "_e2e_oob.wav"
   title = filename + extra
   sf.write(
      
       f"{PREDICTION_PATH}{title}",
       output_wav,
       fs,
       "PCM_16",
   )


   print(f"Saved output for E2E OOB Model!! Took {round(time.time() - start_time, 2)} seconds!")


  




   pred_text, *_ = asr_model(output_wav)[0]


   return pred_text




def forwadrd_casc_model(speech2text_model, s2l, text2speech_model, asr_model, filename, use_finetuned = False, LORA_TARGET = None, finetuned_path = None):
   a = 0
   start_time = time.time()

   if use_finetuned == True:
       create_lora_adapter(speech2text_model.s2t_model, target_modules=LORA_TARGET, rank = 4)
       speech2text_model.s2t_model.eval()
       speech2text_model.s2t_model.load_state_dict(torch.load(finetuned_path))


   spembs = None
   if text2speech_model.use_spembs:
       xvector_ark = [p for p in glob.glob(f"tts_multi-speaker_model/dump/**/spk_xvector.ark", recursive=True) if "tr" in p][0]
       xvectors = {k: v for k, v in kaldiio.load_ark(xvector_ark)}
       spks = list(xvectors.keys())


       random_spk_idx = np.random.randint(0, len(spks))
       spk = spks[random_spk_idx]
       spembs = xvectors[spk]
       print(f"selected spk, x-vector: {spk}")
   sids = None
   if text2speech_model.use_sids:
       spk2sid = glob.glob(f"tts_multi-speaker_model/dump/**/spk2sid", recursive=True)[0]
       with open(spk2sid) as f:
           lines = [line.strip() for line in f.readlines()]
       sid2spk = {int(line.split()[1]): line.split()[0] for line in lines}


       sids = np.array(np.random.randint(1, len(sid2spk)))
       print (sids)
       spk = "p237"
       print(f"selected spk, speaker-id: {spk}")
   speech = None
   if text2speech_model.use_speech:
       speech = torch.randn(50000,) * 0.01
  
   iso_codes = ['abk', 'afr', 'amh', 'ara', 'asm', 'ast', 'aze', 'bak', 'bas', 'bel', 'ben', 'bos', 'bre', 'bul', 'cat', 'ceb', 'ces', 'chv', 'ckb', 'cmn', 'cnh', 'cym', 'dan', 'deu', 'dgd', 'div', 'ell', 'eng', 'epo', 'est', 'eus', 'fas', 'fil', 'fin', 'fra', 'frr', 'ful', 'gle', 'glg', 'grn', 'guj', 'hat', 'hau', 'heb', 'hin', 'hrv', 'hsb', 'hun', 'hye', 'ibo', 'ina', 'ind', 'isl', 'ita', 'jav', 'jpn', 'kab', 'kam', 'kan', 'kat', 'kaz', 'kea', 'khm', 'kin', 'kir', 'kmr', 'kor', 'lao', 'lav', 'lga', 'lin', 'lit', 'ltz', 'lug', 'luo', 'mal', 'mar', 'mas', 'mdf', 'mhr', 'mkd', 'mlt', 'mon', 'mri', 'mrj', 'mya', 'myv', 'nan', 'nep', 'nld', 'nno', 'nob', 'npi', 'nso', 'nya', 'oci', 'ori', 'orm', 'ory', 'pan', 'pol', 'por', 'pus', 'quy', 'roh', 'ron', 'rus', 'sah', 'sat', 'sin', 'skr', 'slk', 'slv', 'sna', 'snd', 'som', 'sot', 'spa', 'srd', 'srp', 'sun', 'swa', 'swe', 'swh', 'tam', 'tat', 'tel', 'tgk', 'tgl', 'tha', 'tig', 'tir', 'tok', 'tpi', 'tsn', 'tuk', 'tur', 'twi', 'uig', 'ukr', 'umb', 'urd', 'uzb', 'vie', 'vot', 'wol', 'xho', 'yor', 'yue', 'zho', 'zul']
   lang_names = ['Abkhazian', 'Afrikaans', 'Amharic', 'Arabic', 'Assamese', 'Asturian', 'Azerbaijani', 'Bashkir', 'Basa (Cameroon)', 'Belarusian', 'Bengali', 'Bosnian', 'Breton', 'Bulgarian', 'Catalan', 'Cebuano', 'Czech', 'Chuvash', 'Central Kurdish', 'Mandarin Chinese', 'Hakha Chin', 'Welsh', 'Danish', 'German', 'Dagaari Dioula', 'Dhivehi', 'Modern Greek (1453-)', 'English', 'Esperanto', 'Estonian', 'Basque', 'Persian', 'Filipino', 'Finnish', 'French', 'Northern Frisian', 'Fulah', 'Irish', 'Galician', 'Guarani', 'Gujarati', 'Haitian', 'Hausa', 'Hebrew', 'Hindi', 'Croatian', 'Upper Sorbian', 'Hungarian', 'Armenian', 'Igbo', 'Interlingua (International Auxiliary Language Association)', 'Indonesian', 'Icelandic', 'Italian', 'Javanese', 'Japanese', 'Kabyle', 'Kamba (Kenya)', 'Kannada', 'Georgian', 'Kazakh', 'Kabuverdianu', 'Khmer', 'Kinyarwanda', 'Kirghiz', 'Northern Kurdish', 'Korean', 'Lao', 'Latvian', 'Lungga', 'Lingala', 'Lithuanian', 'Luxembourgish', 'Ganda', 'Luo (Kenya and Tanzania)', 'Malayalam', 'Marathi', 'Masai', 'Moksha', 'Eastern Mari', 'Macedonian', 'Maltese', 'Mongolian', 'Maori', 'Western Mari', 'Burmese', 'Erzya', 'Min Nan Chinese', 'Nepali (macrolanguage)', 'Dutch', 'Norwegian Nynorsk', 'Norwegian Bokmål', 'Nepali (individual language)', 'Pedi', 'Nyanja', 'Occitan (post 1500)', 'Oriya (macrolanguage)', 'Oromo', 'Odia', 'Panjabi', 'Polish', 'Portuguese', 'Pushto', 'Ayacucho Quechua', 'Romansh', 'Romanian', 'Russian', 'Yakut', 'Santali', 'Sinhala', 'Saraiki', 'Slovak', 'Slovenian', 'Shona', 'Sindhi', 'Somali', 'Southern Sotho', 'Spanish', 'Sardinian', 'Serbian', 'Sundanese', 'Swahili (macrolanguage)', 'Swedish', 'Swahili (individual language)', 'Tamil', 'Tatar', 'Telugu', 'Tajik', 'Tagalog', 'Thai', 'Tigre', 'Tigrinya', 'Toki Pona', 'Tok Pisin', 'Tswana', 'Turkmen', 'Turkish', 'Twi', 'Uighur', 'Ukrainian', 'Umbundu', 'Urdu', 'Uzbek', 'Vietnamese', 'Votic', 'Wolof', 'Xhosa', 'Yoruba', 'Yue Chinese', 'Chinese', 'Zulu']


   task_codes = ['asr', 'st_ara', 'st_cat', 'st_ces', 'st_cym', 'st_deu', 'st_eng', 'st_est', 'st_fas', 'st_fra', 'st_ind', 'st_ita', 'st_jpn', 'st_lav', 'st_mon', 'st_nld', 'st_por', 'st_ron', 'st_rus', 'st_slv', 'st_spa', 'st_swe', 'st_tam', 'st_tur', 'st_vie', 'st_zho']
   task_names = ['Automatic Speech Recognition', 'Translate to Arabic', 'Translate to Catalan', 'Translate to Czech', 'Translate to Welsh', 'Translate to German', 'Translate to English', 'Translate to Estonian', 'Translate to Persian', 'Translate to French', 'Translate to Indonesian', 'Translate to Italian', 'Translate to Japanese', 'Translate to Latvian', 'Translate to Mongolian', 'Translate to Dutch', 'Translate to Portuguese', 'Translate to Romanian', 'Translate to Russian', 'Translate to Slovenian', 'Translate to Spanish', 'Translate to Swedish', 'Translate to Tamil', 'Translate to Turkish', 'Translate to Vietnamese', 'Translate to Chinese']


   lang2code = dict(
       [('Unknown', 'none')] + sorted(list(zip(lang_names, iso_codes)), key=lambda x: x[0])
   )
   task2code = dict(sorted(list(zip(task_names, task_codes)), key=lambda x: x[0]))


   code2lang = dict([(v, k) for k, v in lang2code.items()])


   src_lang = "Spanish"
   task = "Translate to English"
   beam_size = 5
   long_form = False
   text_prev = ""
   task_sym = f'<{task2code[task]}>'
   lang_code = lang2code[src_lang]
   if lang_code == 'none':
       lang_code = s2l(speech)[0][0].strip()[1:-1]
   lang_sym = f'<{lang_code}>'
   speech2text_model.beam_search.beam_size = int(beam_size)




   speech, rate = librosa.load(f"{SOURCE_PATH}{filename}", sr=16000) # speech has shape (len,); resample to 16k Hz


   speech2text_model.maxlenratio = -min(300, int((len(speech) / rate) * 10))  # assuming 10 tokens per second
   translated_text = speech2text_model(speech, text_prev, lang_sym=lang_sym, task_sym=task_sym)[0][-2]




   with torch.no_grad():
       output_wav = text2speech_model(translated_text, speech=speech, spembs=spembs, sids=sids)["wav"].cpu()


   if use_finetuned == False:
       extra = "_casc_oob.wav"
   else:
       extra = "_casv_finetuned.wav"
   title = filename + extra
   sf.write(f"{PREDICTION_PATH}{title}", output_wav.numpy(), text2speech_model.fs, "PCM_16")


   processed_speech, rate = sf.read(f"{PREDICTION_PATH}{title}")
   processed_speech = librosa.resample(processed_speech, rate, 16000)


   pred_text, *_ = asr_model(processed_speech)[0]


   if use_finetuned == False:
       print(f"Saved output for Cascaded OOB Model!! Took {round(time.time() - start_time, 2)} seconds!")
   else:
       print(f"Saved output for Cascaded Finetuned Model!! Took {round(time.time() - start_time, 2)} seconds!")
  
   return pred_text






def main():
   a = 0
   # Metrics Models!
  
   bleu = BLEU(effective_order=True)
   blaser = load_blaser_model("blaser_2_0_qe").eval()
   text_embedder = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder")
   meteor = evaluate.load('meteor')
   comet = evaluate.load('comet')
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
   # Speech2Speech Model
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


   # TTS Model
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


   # S2T Model
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
                                     filename=filename)


   pred_text_casc_oob = forwadrd_casc_model(speech2text_model=speech2text_model,
                                            s2l=s2l,
                                            text2speech_model=text2speech,
                                            asr_model=asr_model,
                                            filename=filename,
                                            use_finetuned=False)
  
   pred_text_casc_fine = forwadrd_casc_model(speech2text_model=speech2text_model,
                                            s2l=s2l,
                                            text2speech_model=text2speech,
                                            asr_model=asr_model,
                                            filename=filename,
                                            use_finetuned=True,
                                            LORA_TARGET=LORA_TARGET,
                                            finetuned_path=FINETUNED_MODEL_PATH)
  
   print("Step 3 DONE: Performed Forward feeding")
   prediction_texts = [pred_text_e2e, pred_text_casc_oob, pred_text_casc_fine]
   source_texts = [SOURCE_TEXT for _ in range(3)]
   reference_texts = [REF_TEXT for _ in range(3)]




   # EVALUATION
   list_asr_bleu, list_comet, list_meteor, list_blaser = get_all_metrics(source_texts=source_texts,
                   pred_texts=prediction_texts,
                   ref_texts=reference_texts,
                   bleu=bleu,
                   blaser=blaser,
                   text_embedder=text_embedder,
                   meteor=meteor,
                   comet=comet)
  
   list_bleu = [a for a, _, _ in list_asr_bleu]
   list_bp = [b for _, b, _ in list_asr_bleu]
   list_ratio = [c for _, _, c in list_asr_bleu]
  


   # SAVE RESULTS
   df_final = pd.DataFrame({"Filename": [filename for _ in range(3)],
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
   start_time = time.time() 
   main()
   print()
   print(f"Total Time = {round(time.time() - start_time, 2)} seconds!")
   

