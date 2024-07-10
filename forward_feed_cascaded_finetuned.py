
import numpy as np
import pandas as pd
import espnetez as ez

import torch
import gradio as gr
import librosa
import os
from espnet2.bin.s2t_inference_language import Speech2Language
from espnet2.bin.s2t_inference import Speech2Text

if not torch.cuda.is_available():
    raise RuntimeError("Please use GPU for better inference speed.")
import glob
import os
import kaldiio
import string

from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
from sacrebleu.metrics import BLEU

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

bleu = BLEU(effective_order=True)

ASR_SCORES = []
SOURCE_PATH = "/data/shire/data/aaditd/speech_data/source_dataset/clips_petite/"
GOLD_PATH = "/data/shire/data/aaditd/speech_data/target_dataset/dev.tsv"
df = pd.read_csv(GOLD_PATH, header=None)

Gold_Dict = {}
count = 0
for _, row in df.iterrows():
  wav_file, ref_text = row[0].split("\t")
  Gold_Dict[wav_file] = ref_text
  
print(len(Gold_Dict))
# Datasets library


text2speech = Text2Speech.from_pretrained(
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

# Get model directory path
from espnet_model_zoo.downloader import ModelDownloader
d = ModelDownloader()
# model_dir = os.path.dirname(d.download_and_unpack(tag)["train_config"])

# X-vector selection
spembs = None
if text2speech.use_spembs:
    xvector_ark = [p for p in glob.glob(f"tts_multi-speaker_model/dump/**/spk_xvector.ark", recursive=True) if "tr" in p][0]
    xvectors = {k: v for k, v in kaldiio.load_ark(xvector_ark)}
    spks = list(xvectors.keys())

    # randomly select speaker
    random_spk_idx = np.random.randint(0, len(spks))
    spk = spks[random_spk_idx]
    spembs = xvectors[spk]
    print(f"selected spk, x-vector: {spk}")

# Speaker ID selection
sids = None
if text2speech.use_sids:
    spk2sid = glob.glob(f"tts_multi-speaker_model/dump/**/spk2sid", recursive=True)[0]
    with open(spk2sid) as f:
        lines = [line.strip() for line in f.readlines()]
    sid2spk = {int(line.split()[1]): line.split()[0] for line in lines}

    # randomly select speaker
    sids = np.array(np.random.randint(1, len(sid2spk)))
    print (sids)
    # good sid values: 70, p246, p334, p239, p260, good looking man: p237
    #spk = sid2spk[int(sids)]
    spk = "p237"
    print(f"selected spk, speaker-id: {spk}")

# Reference speech selection for GST
speech = None
if text2speech.use_speech:
    # you can change here to load your own reference speech
    # e.g.
    # import soundfile as sf
    # speech, fs = sf.read("/path/to/reference.wav")
    # speech = torch.from_numpy(speech).float()
    speech = torch.randn(50000,) * 0.01

model_tag = "espnet/owsm_v3.1_ebf"
device = "cuda"

s2l = Speech2Language.from_pretrained(
    model_tag=model_tag,
    device=device,
    nbest=1,
)

model = Speech2Text.from_pretrained(
    model_tag=model_tag,
    device=device,
    beam_size=5,
    ctc_weight=0.0,
    maxlenratio=0.0,
    # below are default values which can be overwritten in __call__
    lang_sym="<eng>",
    task_sym="<asr>",
    predict_time=False
)


def text_normalizer(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

from espnet2.layers.create_adapter_fn import create_lora_adapter
LORA_TARGET = ["w_1", "w_2", "merge_proj"]

create_lora_adapter(model.s2t_model, target_modules=LORA_TARGET, rank = 4) # not needed if you just want to run inference with pre-trained model
model.s2t_model.eval()



FINETUNED_MODEL_PATH = "/data/shire/data/aaditd/speech_data/exp_learning_rate_1e-7/1epoch.pth"


d = torch.load(FINETUNED_MODEL_PATH) # not needed if you just want to run inference with pre-trained model
model.s2t_model.load_state_dict(d) # not needed if you just want to run inference with pre-trained model

iso_codes = ['abk', 'afr', 'amh', 'ara', 'asm', 'ast', 'aze', 'bak', 'bas', 'bel', 'ben', 'bos', 'bre', 'bul', 'cat', 'ceb', 'ces', 'chv', 'ckb', 'cmn', 'cnh', 'cym', 'dan', 'deu', 'dgd', 'div', 'ell', 'eng', 'epo', 'est', 'eus', 'fas', 'fil', 'fin', 'fra', 'frr', 'ful', 'gle', 'glg', 'grn', 'guj', 'hat', 'hau', 'heb', 'hin', 'hrv', 'hsb', 'hun', 'hye', 'ibo', 'ina', 'ind', 'isl', 'ita', 'jav', 'jpn', 'kab', 'kam', 'kan', 'kat', 'kaz', 'kea', 'khm', 'kin', 'kir', 'kmr', 'kor', 'lao', 'lav', 'lga', 'lin', 'lit', 'ltz', 'lug', 'luo', 'mal', 'mar', 'mas', 'mdf', 'mhr', 'mkd', 'mlt', 'mon', 'mri', 'mrj', 'mya', 'myv', 'nan', 'nep', 'nld', 'nno', 'nob', 'npi', 'nso', 'nya', 'oci', 'ori', 'orm', 'ory', 'pan', 'pol', 'por', 'pus', 'quy', 'roh', 'ron', 'rus', 'sah', 'sat', 'sin', 'skr', 'slk', 'slv', 'sna', 'snd', 'som', 'sot', 'spa', 'srd', 'srp', 'sun', 'swa', 'swe', 'swh', 'tam', 'tat', 'tel', 'tgk', 'tgl', 'tha', 'tig', 'tir', 'tok', 'tpi', 'tsn', 'tuk', 'tur', 'twi', 'uig', 'ukr', 'umb', 'urd', 'uzb', 'vie', 'vot', 'wol', 'xho', 'yor', 'yue', 'zho', 'zul']
lang_names = ['Abkhazian', 'Afrikaans', 'Amharic', 'Arabic', 'Assamese', 'Asturian', 'Azerbaijani', 'Bashkir', 'Basa (Cameroon)', 'Belarusian', 'Bengali', 'Bosnian', 'Breton', 'Bulgarian', 'Catalan', 'Cebuano', 'Czech', 'Chuvash', 'Central Kurdish', 'Mandarin Chinese', 'Hakha Chin', 'Welsh', 'Danish', 'German', 'Dagaari Dioula', 'Dhivehi', 'Modern Greek (1453-)', 'English', 'Esperanto', 'Estonian', 'Basque', 'Persian', 'Filipino', 'Finnish', 'French', 'Northern Frisian', 'Fulah', 'Irish', 'Galician', 'Guarani', 'Gujarati', 'Haitian', 'Hausa', 'Hebrew', 'Hindi', 'Croatian', 'Upper Sorbian', 'Hungarian', 'Armenian', 'Igbo', 'Interlingua (International Auxiliary Language Association)', 'Indonesian', 'Icelandic', 'Italian', 'Javanese', 'Japanese', 'Kabyle', 'Kamba (Kenya)', 'Kannada', 'Georgian', 'Kazakh', 'Kabuverdianu', 'Khmer', 'Kinyarwanda', 'Kirghiz', 'Northern Kurdish', 'Korean', 'Lao', 'Latvian', 'Lungga', 'Lingala', 'Lithuanian', 'Luxembourgish', 'Ganda', 'Luo (Kenya and Tanzania)', 'Malayalam', 'Marathi', 'Masai', 'Moksha', 'Eastern Mari', 'Macedonian', 'Maltese', 'Mongolian', 'Maori', 'Western Mari', 'Burmese', 'Erzya', 'Min Nan Chinese', 'Nepali (macrolanguage)', 'Dutch', 'Norwegian Nynorsk', 'Norwegian Bokmål', 'Nepali (individual language)', 'Pedi', 'Nyanja', 'Occitan (post 1500)', 'Oriya (macrolanguage)', 'Oromo', 'Odia', 'Panjabi', 'Polish', 'Portuguese', 'Pushto', 'Ayacucho Quechua', 'Romansh', 'Romanian', 'Russian', 'Yakut', 'Santali', 'Sinhala', 'Saraiki', 'Slovak', 'Slovenian', 'Shona', 'Sindhi', 'Somali', 'Southern Sotho', 'Spanish', 'Sardinian', 'Serbian', 'Sundanese', 'Swahili (macrolanguage)', 'Swedish', 'Swahili (individual language)', 'Tamil', 'Tatar', 'Telugu', 'Tajik', 'Tagalog', 'Thai', 'Tigre', 'Tigrinya', 'Toki Pona', 'Tok Pisin', 'Tswana', 'Turkmen', 'Turkish', 'Twi', 'Uighur', 'Ukrainian', 'Umbundu', 'Urdu', 'Uzbek', 'Vietnamese', 'Votic', 'Wolof', 'Xhosa', 'Yoruba', 'Yue Chinese', 'Chinese', 'Zulu']

task_codes = ['asr', 'st_ara', 'st_cat', 'st_ces', 'st_cym', 'st_deu', 'st_eng', 'st_est', 'st_fas', 'st_fra', 'st_ind', 'st_ita', 'st_jpn', 'st_lav', 'st_mon', 'st_nld', 'st_por', 'st_ron', 'st_rus', 'st_slv', 'st_spa', 'st_swe', 'st_tam', 'st_tur', 'st_vie', 'st_zho']
task_names = ['Automatic Speech Recognition', 'Translate to Arabic', 'Translate to Catalan', 'Translate to Czech', 'Translate to Welsh', 'Translate to German', 'Translate to English', 'Translate to Estonian', 'Translate to Persian', 'Translate to French', 'Translate to Indonesian', 'Translate to Italian', 'Translate to Japanese', 'Translate to Latvian', 'Translate to Mongolian', 'Translate to Dutch', 'Translate to Portuguese', 'Translate to Romanian', 'Translate to Russian', 'Translate to Slovenian', 'Translate to Spanish', 'Translate to Swedish', 'Translate to Tamil', 'Translate to Turkish', 'Translate to Vietnamese', 'Translate to Chinese']

lang2code = dict(
    [('Unknown', 'none')] + sorted(list(zip(lang_names, iso_codes)), key=lambda x: x[0])
)
task2code = dict(sorted(list(zip(task_names, task_codes)), key=lambda x: x[0]))

code2lang = dict([(v, k) for k, v in lang2code.items()])


# Copied from Whisper utils
def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def predict(audio_path, src_lang: str, task: str, beam_size, long_form: bool, text_prev: str,):
    task_sym = f'<{task2code[task]}>'
    model.beam_search.beam_size = int(beam_size)

    # Our model is trained on 30s and 16kHz
    speech, rate = librosa.load(audio_path, sr=16000) # speech has shape (len,); resample to 16k Hz

    lang_code = lang2code[src_lang]
    if lang_code == 'none':
        # Detect language using the first 30s of speech
        lang_code = s2l(speech)[0][0].strip()[1:-1]
    lang_sym = f'<{lang_code}>'

    # ASR or ST
    if long_form:
        try:
            model.maxlenratio = -300
            utts = model.decode_long(
                speech,
                condition_on_prev_text=False,
                init_text=text_prev,
                end_time_threshold="<29.00>",
                lang_sym=lang_sym,
                task_sym=task_sym,
            )

            text = []
            for t1, t2, res in utts:
                text.append(f"[{format_timestamp(seconds=t1)} --> {format_timestamp(seconds=t2)}] {res}")
            text = '\n'.join(text)

            return code2lang[lang_code], text
        except:
            print("An exception occurred in long-form decoding. Fall back to standard decoding (only first 30s)")

    model.maxlenratio = -min(300, int((len(speech) / rate) * 10))  # assuming 10 tokens per second
    text = model(speech, text_prev, lang_sym=lang_sym, task_sym=task_sym)[0][-2]

    return code2lang[lang_code], text
import pandas as pd

id_start = len("common_voice_en_")
def has_id(file, id):
    return file[id_start:id_start+len(id)] == id
import soundfile as sf
#iterate through files in the clips folder

df = pd.read_csv("/data/shire/data/aaditd/speech_data/source_dataset/dev.tsv", sep = '\t')
dev_files = list(df["path"])

i = 0
count = 0

output_csv = {"Prediction": [], "Gold":[], "ASR_BLEU":[], "File": []}
# for filename, ref_text in Gold_Dict.items():
for filename in os.listdir(SOURCE_PATH):

    ref_text = Gold_Dict[filename]

        
    _, text = predict(f"{SOURCE_PATH}{filename}", "Spanish", "Translate to English", 5, False,"")
    
    with torch.no_grad():
        wav = text2speech(text, speech=speech, spembs=spembs, sids=sids)["wav"].cpu()

    # print(f"Writing at fs = {text2speech.fs}")
    # exit()
    sf.write(f"/data/shire/data/aaditd/speech_data/pred_fine_casc_1e-7/{filename}.wav", wav.numpy(), text2speech.fs, "PCM_16")


    processed_speech, rate = sf.read(f"/data/shire/data/aaditd/speech_data/pred_fine_casc_1e-7/{filename}.wav")
    # assert rate == 16000, f"The actual rate is: {rate}!!!!!"
    processed_speech = librosa.resample(processed_speech, rate, 16000)



    text, *_ = asr_model(processed_speech)[0]
    
    score = bleu.sentence_score(text_normalizer(text), [text_normalizer(ref_text)])

    output_csv["Prediction"].append(text_normalizer(text))
    output_csv["Gold"].append(text_normalizer(ref_text))
    output_csv['ASR_BLEU'].append(score)
    output_csv['File'].append(filename)
    # print(f"SCORE: {score} | PRED: {text_normalizer(text)} | GOLD: {text_normalizer(ref_text)} ")





    count += 1
    # print(f"File {filename} written!")

    if count % 100 == 0:
        print(f"        {count} DONE!")
        df2 = pd.DataFrame(output_csv)
        df2.to_csv(f"/home/aaditd/2_Speech_Project/results/casc_finetuned_1e-7_1_epoch_results/finetuned_output_{count}.csv")

    
    
