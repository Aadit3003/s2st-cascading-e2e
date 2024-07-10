""" This module contains the code to fine-tune the Cascaded S2ST system on the
CVSS-C dataset.

This is done prior to running the forward_feed_cascaded_finetuned.py script, to 
ensure a fairer comparison with the end-to-end model (pre-trained on CVSS-C).
"""
import pandas as pd
import numpy as np

import torch
import espnetez as ez
from espnet2.bin.s2t_inference_language import Speech2Language
from espnet2.bin.s2t_inference import Speech2Text
from espnet2.layers.create_adapter_fn import create_lora_adapter
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
from datasets import load_dataset, Audio

text_file = "/data/shire/data/aaditd/speech_data/source_dataset/covost_v2.es_en.tsv"
#load text file into a pandas dataframe with path as the id

text_df = pd.read_csv(text_file, sep="\t")
text_df = text_df.set_index("path")
#print out "common_voice_es_19600147.mp3"
# text_df.loc["common_voice_es_19600147.mp3"]['translation']


"""EVAN LOOK HERE:|{'audio': {'path': '/home/efellman/speech_proj/data/dev/common_voice_es_18308858.mp3.wav', 'array': array([ 2.97360644e-03,  4.32699605e-03,  4.34680204e-03, ...,
       -8.11031307e-10,  5.98494003e-09, -8.77921745e-09]), 'sampling_rate': 16000}, 'label': None}|"""
print("Loading Dataset")

train_dataset = load_dataset("audiofolder", data_dir=f"/data/shire/data/aaditd/speech_data/source_dataset/clips_16", split="train[:800]")
valid_dataset = load_dataset("audiofolder", data_dir=f"/data/shire/data/aaditd/speech_data/source_dataset/clips_16", split="train[800:1000]")
print("Finished loading data.")
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
valid_dataset = valid_dataset.cast_column("audio", Audio(sampling_rate=16000))

def path_to_path(p):
    # print(p)
    # exit(0)
    return p[len("/data/shire/data/aaditd/speech_data/source_dataset/clips_16/"):]

data_info = {
    "speech": lambda d: d['audio']['array'].astype(np.float32),
    "text": lambda d: tokenize(f"<eng><asr><notimestamps> {text_df.loc[path_to_path(d['audio']['path'])]['translation']}"),
    "text_prev": lambda d: tokenize("<na>"),
    "text_ctc": lambda d: tokenize(text_df.loc[path_to_path(d['audio']['path'])]['translation']),
}


def tokenize(text):
    return np.array(converter.tokens2ids(tokenizer.text2tokens(text)))


if not isinstance(train_dataset, ez.dataset.ESPnetEZDataset):
  train_dataset = ez.dataset.ESPnetEZDataset(train_dataset, data_info=data_info)
  valid_dataset = ez.dataset.ESPnetEZDataset(valid_dataset, data_info=data_info)
print("Finished preparing data.")


if not torch.cuda.is_available():
    raise RuntimeError("Please use GPU for better inference speed.")





model_tag = "espnet/owsm_v3.1_ebf"
device = "cuda"

print("Loading Pre-trained model")
pretrained_model = Speech2Text.from_pretrained(
    model_tag=model_tag,
    device=device,
    beam_size=5,
    ctc_weight=0.0,
    maxlenratio=0.0,
    # below are default values which can be overwritten in __call__
    lang_sym="<eng>",
    task_sym="<asr>",
    predict_time=False,
)

pretrain_config = vars(pretrained_model.s2t_train_args)
finetune_config = ez.config.update_finetune_config(
	's2t',
	pretrain_config,
	f"lora_config.yaml"
)


pretrain_config = vars(pretrained_model.s2t_train_args)
tokenizer = pretrained_model.tokenizer
converter = pretrained_model.converter
del pretrained_model

print("Loaded configs.")


def freeze_parameters(model):
    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad = False

def build_model_fn(args):
    pretrained_model = Speech2Text.from_pretrained(
        model_tag=model_tag,
        device=device,
        beam_size=10,
        ctc_weight=0.0,
        maxlenratio=0.0,
        # below are default values which can be overwritten in __call__
        lang_sym="<eng>",
        task_sym="<asr>",
        predict_time=False,
    )
    model = pretrained_model.s2t_model
    model.train()
    # apply lora
    freeze_parameters(model)
    create_lora_adapter(model, rank=4, target_modules=["w_1", "w_2", "merge_proj"])

    #we now have model which is the pytorch model of speech to text.

    #Below we create text2speech. A pytorch model to convert text to speech

    #We take the argmax across each of the tokens 

    return model


trainer = ez.Trainer(
    task='s2t',
    lr=1e-7,
    train_config=finetune_config,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    build_model_fn=build_model_fn, # provide the pre-trained model
    data_info=data_info,
    output_dir="./exp",
    stats_dir="./stats",
    ngpu=1
)
trainer.collect_stats()
trainer.train()

