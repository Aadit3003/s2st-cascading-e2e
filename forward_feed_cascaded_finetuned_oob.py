""" This is one of the main modules and contains the code to forward-feed 
the Cascaded S2ST model (the model fine-tuned on CVSS-C).

These functions are used to generate the translation files for the dev set
while comparing to the end-to-end system. They are also reused in the 
live_s2st_demontration module.
"""
import argparse
import numpy as np
import pandas as pd
import glob
import os
import string
import time

import soundfile as sf
import kaldiio
import gradio as gr
import librosa
import torch
import espnetez as ez
from espnet2.bin.s2t_inference_language import Speech2Language
from espnet2.bin.s2t_inference import Speech2Text
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
from sacrebleu.metrics import BLEU
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text as asr
from espnet2.layers.create_adapter_fn import create_lora_adapter
from espnet_model_zoo.downloader import ModelDownloader
from sacrebleu.metrics import BLEU
import evaluate
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.models.blaser.loader import load_blaser_model
from sacrebleu.metrics import BLEU

from utils.macro_average_results import macro_average_translation_metrics
from expanded_translation_metrics import generate_metrics_for_file_vectorized

DATA_DIRECTORY = "/data/shire/data/aaditd/"

SOURCE_PATH = f"{DATA_DIRECTORY}speech_data/source_dataset/clips_petite/"
PREDICTION_PATH_FINETUNED = f"{DATA_DIRECTORY}speech_data/pred_fine_casc_1e-7/"
PREDICTION_PATH_OOB = f"{DATA_DIRECTORY}speech_data/pred_oob_casc/"

DEV_TARGET_DATASET_PATH = "./dev_dataset/dev_target.tsv" # Gold Reference
DEV_SOURCE_DATASET_PATH = "./dev_dataset/dev_source.tsv"
FINETUNED_MODEL_PATH = f"{DATA_DIRECTORY}speech_data/exp_learning_rate_1e-7/1epoch.pth"

def text_normalizer(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

def forwadrd_cascaded_model(speech2text_model, 
                        speech2language, 
                        text2speech_model, 
                        asr_model, 
                        current_filename, 
                        use_finetuned = False, 
                        lora_target = None, 
                        finetuned_s2t_model_path = None):

   start_time = time.time()

   if use_finetuned == True:
       create_lora_adapter(speech2text_model.s2t_model, target_modules=lora_target, rank = 4)
       speech2text_model.s2t_model.eval()
       speech2text_model.s2t_model.load_state_dict(torch.load(finetuned_s2t_model_path))
       
       PREDICTION_PATH = PREDICTION_PATH_FINETUNED
   else:
       PREDICTION_PATH = PREDICTION_PATH_OOB


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
       lang_code = speech2language(speech)[0][0].strip()[1:-1]
   lang_sym = f'<{lang_code}>'
   speech2text_model.beam_search.beam_size = int(beam_size)




   speech, rate = librosa.load(f"{SOURCE_PATH}{current_filename}", sr=16000) # speech has shape (len,); resample to 16k Hz
   speech2text_model.maxlenratio = -min(300, int((len(speech) / rate) * 10))  # assuming 10 tokens per second
   
   translated_text = speech2text_model(speech, text_prev, lang_sym=lang_sym, task_sym=task_sym)[0][-2]


   with torch.no_grad():
       output_wav = text2speech_model(translated_text, speech=speech, spembs=spembs, sids=sids)["wav"].cpu()


   if use_finetuned == False:
       extra = "_casc_oob.wav"
   else:
       extra = "_casv_finetuned.wav"
   title = current_filename + extra
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
    # Command Line choice of Finetuned vs. OOB model
    parser = argparse.ArgumentParser()
    parser.add_argument("cascaded_model_inference_mode", type=str, help='finetuned or oob')
    args = parser.parse_args()
    cascaded_model_inference_mode = args.cascaded_model_inference_mode
    if cascaded_model_inference_mode == "finetuned":
        use_finetuned = True
    else:
        use_finetuned = False
        
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

    text2speech_instance = Text2Speech.from_pretrained(
        # train_config="tts_multi-speaker_model/exp/tts_train_full_band_multi_spk_vits_raw_phn_tacotron_g2p_en_no_space/config.yaml",
        train_config = "/home/aaditd/2_Speech_Project/tts_config_old.yaml",
        model_file="tts_multi-speaker_model/exp/tts_train_full_band_multi_spk_vits_raw_phn_tacotron_g2p_en_no_space/train.total_count.ave_10best.pth",
        vocoder_tag=str_or_none("none"),
        device="cuda",
        # Only for Tacotron 2 & Transformer
        threshold=0.5,
        # Only for Tacotron 2
        minlenratio=0.0,
        maxlenratio=10.0,
        use_att_constraint=False,
        backward_window=1,
        forward_window=3,
        # Only for FastSpeech & FastSpeech2 & VITS
        speed_control_alpha=1.0,
        # Only for VITS
        noise_scale=0.333,
        noise_scale_dur=0.333,
    )

    model_tag = "espnet/owsm_v3.1_ebf"
    device = "cuda"

    s2l = Speech2Language.from_pretrained(
        model_tag=model_tag,
        device=device,
        nbest=1,
    )

    speech2text_instance = Speech2Text.from_pretrained(
        model_tag=model_tag,
        device=device,
        beam_size=5,
        ctc_weight=0.0,
        maxlenratio=0.0,
        lang_sym="<eng>",
        task_sym="<asr>",
        predict_time=False
    )

    LORA_TARGET = ["w_1", "w_2", "merge_proj"]
    
    # Iterate through the Dev dataset files
    source_df = pd.read_csv(DEV_SOURCE_DATASET_PATH, sep = '\t')

    output_csv = {"Prediction": [], "Gold":[], "ASR_BLEU":[], "File": []}

    count = 0
    for filename in list(source_df["path"]): # Alternatively, use os.listdir(SOURCE_PATH)

        ref_text = Gold_Dict[filename]
        
    # Forward-feed the model
        pred_text = forwadrd_cascaded_model(speech2text_model = speech2text_instance, 
                        speech2language = s2l, 
                        text2speech_model = text2speech_instance, 
                        asr_model = asr_instance, 
                        current_filename = filename, 
                        use_finetuned = use_finetuned, 
                        lora_target = LORA_TARGET, 
                        finetuned_s2t_model_path = FINETUNED_MODEL_PATH)

        output_csv["pred_text"].append(text_normalizer(pred_text))
        output_csv["ref_text"].append(text_normalizer(ref_text))
        output_csv['file'].append(filename)
        
        count += 1
        if count % 100 == 0: # Save every 100 iterations (optional)
            print(f"        {count} DONE!")
            df2 = pd.DataFrame(output_csv)
            df2.to_csv(f"./results/casc_{cascaded_model_inference_mode}_results/casc_output_{count}.csv")


    results_df = pd.DataFrame(output_csv)
    RESULTS_PATH = f"./results/casc_{cascaded_model_inference_mode}_results/casc_output_final.csv"
    results_df.to_csv(RESULTS_PATH)
    
    # Calculate the other Translation metrics
    
    bleu_metric = BLEU(effective_order=True)
    blaser_metric = load_blaser_model("blaser_2_0_qe").eval()
    text_embedder = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder")
    meteor_metric = evaluate.load('meteor')
    comet_metric = evaluate.load('comet')
    
    expanded_results_df = generate_metrics_for_file_vectorized(filename=RESULTS_PATH, 
                                                               bleu=bleu_metric,
                                                               comet=comet_metric,
                                                               meteor=meteor_metric,
                                                               blaser=blaser_metric, 
                                                               text_embedder=text_embedder)
    
    # Write the Macro-Averaged metrics to a results file
    MACRO_RESULTS_PATH = f"./results/casc_{cascaded_model_inference_mode}_results/macro_avg_metrics.txt"
    avg_asr, avg_bp, avg_ratio, avg_comet, avg_meteor, avg_blaser = macro_average_translation_metrics(expanded_results_df)
    with open(MACRO_RESULTS_PATH, "w") as file:
        file.write(f"ASR BLEU: {avg_asr}\n")
        file.write(f"BP: {avg_bp}\n")
        file.write(f"HRR: {avg_ratio}\n")
        file.write(f"COMET: {avg_comet}\n")
        file.write(f"METEOR: {avg_meteor}\n")
        file.write(f"BLASER 2.0: {avg_blaser}\n")

    print("DONE!!")
    
if __name__ == "__main__":
    
    if not torch.cuda.is_available():
        raise RuntimeError("Please use GPU for better inference speed.")
    
    main()
    
