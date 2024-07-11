""" This module contains the code to fine-tune the Speech2Text component of the 
Cascaded S2ST system on the CVSS-C dataset (i.e. the CoVoST 2 dataset).

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

MODEL_TAG = "espnet/owsm_v3.1_ebf"
DEVICE = "cuda"

COVOST2_DATASET_PATH = "/data/shire/data/aaditd/speech_data/source_dataset/covost_v2.es_en.tsv"

def path_to_path(original_path):
    """
    Used in the create_espnet_datasets() function while we want to extract the 
    file name without the entire directory name/

    Args:
        ooriginal_path: The full path name. E.g. /data/shire/data/aaditd/speech_data/source_dataset/clips_16/common_voice_es_18308858.mp3.wav

    Returns:
        Only the file name. E.g. common_voice_es_18308858.mp3.wav
    """
    return original_path[len("/data/shire/data/aaditd/speech_data/source_dataset/clips_16/"):]

def tokenize(tokenizer, converter, text):
    """
    Returns the tokenized text as an np.array.

    Args:
        tokenizer: The tokenizer derived from the pretrained S2T model.
        converter: The converter derived from the pretrained S2T model.
        text: The text you want to tokenzie.

    Returns:
        The tokens as an np.array
    """
    return np.array(converter.tokens2ids(tokenizer.text2tokens(text)))

def freeze_parameters(model):
    """ 
    Used in the build_model() function, which is subsequently passed to the trainer
    object. It freezes the parameters of the pretrained model, so that LORA can be used
    for efficient fine-tuning.
    """
    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad = False

def create_espnet_datasets(tokenizer, converter, text_df):
    """
    Creates espnet train and validation datasets to be passed as
    arguments for the trainer object.

    Args:
        tokenizer: The tokenizer derived from the pretrained S2T model.
        converter: The converter derived from the pretrained S2T model.
        text_df: The dataframe containing the CoVoST 2 data to be 
            converted to train and validation datasets

    Returns:
        The train and validation datasets and the data_info directory
    """   
    
    
    """ Sample
    {'audio': {'path': '/data/shire/data/aaditd/speech_data/source_dataset/clips_16/common_voice_es_18308858.mp3.wav', 
               'array': array([ 2.97360644e-03,  4.32699605e-03,  4.34680204e-03, ..., -8.11031307e-10,  5.98494003e-09, -8.77921745e-09]), 
               'sampling_rate': 16000}, 
     'label': None}
    """
    print("Loading Dataset")

    train_dataset = load_dataset("audiofolder", data_dir=f"/data/shire/data/aaditd/speech_data/source_dataset/clips_16", split="train[:800]")
    valid_dataset = load_dataset("audiofolder", data_dir=f"/data/shire/data/aaditd/speech_data/source_dataset/clips_16", split="train[800:1000]")
    print("Finished loading data.")
    
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    valid_dataset = valid_dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    data_info = {
        "speech": lambda d: d['audio']['array'].astype(np.float32),
        "text": lambda d: tokenize(tokenizer, converter, f"<eng><asr><notimestamps> {text_df.loc[path_to_path(d['audio']['path'])]['translation']}"),
        "text_prev": lambda d: tokenize(tokenizer, converter, "<na>"),
        "text_ctc": lambda d: tokenize(tokenizer, converter, text_df.loc[path_to_path(d['audio']['path'])]['translation']),
    }

    if not isinstance(train_dataset, ez.dataset.ESPnetEZDataset):
        train_dataset = ez.dataset.ESPnetEZDataset(train_dataset, data_info=data_info)
        valid_dataset = ez.dataset.ESPnetEZDataset(valid_dataset, data_info=data_info)
    
    
    return train_dataset,valid_dataset,data_info

def build_model_fn(args):
    """
    Returns a pretrained S2T model with its parameters frozen and with its
    LORA adapter. Passed as an argument to the trainer object.
    """
    global MODEL_TAG
    global DEVICE
    
    pretrained_model = Speech2Text.from_pretrained(
        model_tag=MODEL_TAG,
        device=DEVICE,
        beam_size=10,
        ctc_weight=0.0,
        maxlenratio=0.0,
        lang_sym="<eng>",
        task_sym="<asr>",
        predict_time=False,
    )
    model = pretrained_model.s2t_model
    model.train()
    
    # Apply LoRA
    freeze_parameters(model)
    create_lora_adapter(model, rank=4, target_modules=["w_1", "w_2", "merge_proj"])

    return model

def main():
    
    # Step 1) Load the pretrained model and extract its configs!
    pretrained_model = Speech2Text.from_pretrained(
        model_tag=MODEL_TAG,
        device=DEVICE,
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
    
    # Step 2) Load the training (fine-tuning) data and create the dataset using the
    # previously extracted tokenizer from the pretrained model
    text_df = pd.read_csv(COVOST2_DATASET_PATH, sep="\t")
    text_df = text_df.set_index("path")

    train_dataset, valid_dataset, data_info = create_espnet_datasets(tokenizer, converter, text_df)
    
    
    print("Finished preparing data.")

    # Step 3) Create a trainer instance, provide the pre-trained model and call train()
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



if __name__ == "__main__":
    
    if not torch.cuda.is_available():
        raise RuntimeError("Please use GPU for better inference speed.")
    
    main()