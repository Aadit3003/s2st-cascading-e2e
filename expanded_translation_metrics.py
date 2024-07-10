""" This module contains the functions used to compute the following Speech
Translation metrics:- COMET, METEOR, BLASER.

These are in addition to the ASR-BLEU score that is calculated in the forward_feed
or live_s2st_demonstration modules. This module is called after inferencing the S2ST
system and is used to create a csv file with all 4 metrics for each sample in the dev
dataset.
"""
import numpy as np
import pandas as pd
from string import punctuation
import string
import os

from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.models.blaser.loader import load_blaser_model
import evaluate


DEV_SOURCE_PATH = "/home/aaditd/2_Speech_Project/dev_source.tsv"
DEV_TARGET_PATH = "/home/aaditd/2_Speech_Project/dev_target.tsv"
FILE_PATH = "/home/aaditd/2_Speech_Project/metrics/raw/"
OUTPUT_PATH = "/home/aaditd/2_Speech_Project/metrics/scored/"


def text_normalizer(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))



def blaser_score_vectorized(source_texts, pred_texts, blaser, text_embedder):


    src_embs = text_embedder.predict(source_texts, source_lang="spa_Latn") # "Le chat s'assit sur le tapis."
    mt_embs = text_embedder.predict(pred_texts, source_lang="eng_Latn") # "The cat sat down on the carpet."
    return [b.item() for b in list(blaser(src=src_embs, mt=mt_embs))]  # 4.708

def comet_score_vectorized(pred_texts, ref_texts, source_texts, comet): 

    results = comet.compute(predictions=pred_texts, 
                                   references=ref_texts, 
                                   sources=source_texts)
    
    return results["scores"]

# source_dic, target_dic = {}, {}
# reverse_target_dic = {}

# for index, row in dev_source_df.iterrows():
#     source_dic[row['path']] = text_normalizer(row['sentence'])

# for index, row in dev_target_df.iterrows():
    # target_dic[row['path']] = text_normalizer(row['sentence'])

def meteor_score_vectorized(pred_text, ref_text, meteor):

    return meteor.compute(predictions=[pred_text], references=[[ref_text]])['meteor']





def add_source_text_path_to_file(filename, save_path):
    
    dev_source_df = pd.read_csv(DEV_SOURCE_PATH, sep='\t')

    dev_target_df = pd.read_csv(DEV_TARGET_PATH, sep='\t', names=['path', 'sentence'])

    source_dic, target_dic = {}, {}

    paths, sentences = dev_source_df["path"], dev_source_df["sentence"]
    for a, b in zip(paths, sentences):
        source_dic[a] = text_normalizer(b)

    paths, sentences = dev_target_df["path"], dev_target_df["sentence"]
    for a, b in zip(paths, sentences):
        target_dic[a] = text_normalizer(b)
    
    reverse_target_dic = {v:k for k, v in target_dic.items()}
    
    new_dic = {"path": [],
               "source_text": [],
               "pred_text": [],
               "ref_text": [],
               "ASR_BLEU": []}
    df = pd.read_csv(filename)
    # print(df.columns)

    i = 0
    for index, row in df.iterrows():
        # print(i)
        pred_text = row["Prediction"]
        ref_text = row["Gold"]
        asr_bleu = row["ASR_BLEU"]
        path = reverse_target_dic[ref_text]
        source_text = source_dic[path]

        new_dic["path"].append(path)
        new_dic["source_text"].append(source_text)
        new_dic["pred_text"].append(pred_text)
        new_dic["ref_text"].append(ref_text)
        new_dic["ASR_BLEU"].append(asr_bleu)
        i += 1
        
    
    new_df = pd.DataFrame(new_dic)
    title = filename.split("/")[-1]
    new_df.to_csv(OUTPUT_PATH + title)

    print("DONE!")

    print(new_df.head(5))
    
    new_df.to_csv(save_path)
    return new_df

def generate_metrics_for_file_vectorized(filename, comet, meteor, blaser, text_embedder):
    df = pd.read_csv(filename)
    df = df.replace(np.nan, '', regex=True)
    # print(df.columns)

    meteor_list, comet_list, blaser_list = [], [], []

    # i = 0
    # for index, row in df.iterrows():
        
    pred_texts = df["pred_text"]
    ref_texts = df["ref_text"]
    source_texts = df["source_text"]

    comet_list = comet_score_vectorized(pred_texts=pred_texts, ref_texts=ref_texts, source_texts=source_texts, comet=comet)
    meteor_list = [meteor_score_vectorized(p, r, meteor) for p, r in zip(pred_texts, ref_texts)]
    blaser_list = blaser_score_vectorized(source_texts=source_texts, pred_texts=pred_texts, blaser=blaser, text_embedder=text_embedder)

    print("COMET LIST")
    print(comet_list)
    print()

    print("METEOR LIST")
    print(meteor_list)
    print()

    print("BLASER LIST")
    print(blaser_list)
    print()

    df["COMET"] = comet_list
    df["METEOR"] = meteor_list
    df["BLASER"] = blaser_list

    title = filename.split("/")[-1]
    df.to_csv(f"{OUTPUT_PATH}{title}")

    return df





# make_raw_file(FILE_PATH)

def main():
    original_results_filename = "/home/aaditd/2_Speech_Project/results/casc_finetuned_1e-7_1_epoch_results/finetuned_output_1000.csv"
    save_path = "/home/aaditd/2_Speech_Project/metrics/raw/casc_finetuned_1e-7_1_epoch.csv"
    add_source_text_path_to_file(original_results_filename, save_path)
    
    

        
        
    blaser = load_blaser_model("blaser_2_0_qe").eval()
    text_embedder = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder")
    meteor = evaluate.load('meteor')
    comet = evaluate.load('comet')
    
    expanded_filename = save_path
    print(expanded_filename)
    generate_metrics_for_file_vectorized(expanded_filename, comet, meteor, blaser, text_embedder)

    print("DONE!!")

    


if __name__ == "__main__":
    main()

