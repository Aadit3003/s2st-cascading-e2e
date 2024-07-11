""" This module contains the functions used to compute the following Speech
Translation metrics:- ASR BLEU (and BP, HRR), COMET, METEOR, and BLASER 2.0.

This module is called after inferencing the S2ST system and is used to create 
a csv file with all 4 metrics for each sample in the dev dataset.
"""
import numpy as np
import pandas as pd
from string import punctuation
import string
import os

from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.models.blaser.loader import load_blaser_model
import evaluate

from utils.macro_average_results import segment_bleu_score_string


DEV_SOURCE_PATH = "/home/aaditd/2_Speech_Project/dev_source.tsv"
DEV_TARGET_PATH = "/home/aaditd/2_Speech_Project/dev_target.tsv"
FILE_PATH = "/home/aaditd/2_Speech_Project/metrics/raw/"
OUTPUT_PATH = "/home/aaditd/2_Speech_Project/metrics/scored/"

def text_normalizer(text):
    """
    Converts sentences to lower case and removes punctuation.
    Used to normalize source and target sentences for BLEU scoring.
    """
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

def blaser_score_vectorized(source_texts, pred_texts, blaser, text_embedder):
    """
    Computes the BLASER 2.0 score for a list of source texts and corresponding predictions.

    Args:
        source_texts: A list of source language sentences.
        pred_texts: A list of predictions by the S2ST model.
        blaser: The huggingface instance that computes the BLASER 2.0 metric.
        text_embedder: The text embedder model used by the blaser instance.

    Returns:
        The vector of BLASER 2.0 scores for the predictions.
    """
    assert len(source_texts) == len(pred_texts), f"The number of source sentences ({len(source_texts)}) is not the same as the number of predictions ({len(pred_texts)})"
    src_embs = text_embedder.predict(source_texts, source_lang="spa_Latn") # "Le chat s'assit sur le tapis."
    mt_embs = text_embedder.predict(pred_texts, source_lang="eng_Latn") # "The cat sat down on the carpet."
    return [b.item() for b in list(blaser(src=src_embs, mt=mt_embs))]  # 4.708

def comet_score_vectorized(pred_texts, ref_texts, source_texts, comet): 
    """    
    Computes the COMET score for a list of source texts, ref texts, and corresponding predictions.

    Args:
        pred_texts: A list of predictions by the S2ST model.
        ref_texts: A list of target language sentences (reference translations)
        source_texts: A list of source language sentences.
        comet: The huggingface instance that computes the COMET metric

    Returns:
        The vector of COMET scores for the predictions.
    """

    assert len(source_texts) == len(ref_texts), f"The number of source sentences ({len(source_texts)}) is not the same as the number of reference sentences ({len(ref_texts)})"
    assert len(source_texts) == len(pred_texts), f"The number of source sentences ({len(source_texts)}) is not the same as the number of predictions ({len(pred_texts)})"
    results = comet.compute(predictions=pred_texts, 
                                   references=ref_texts, 
                                   sources=source_texts)
    
    return results["scores"]

def meteor_score_vectorized(pred_texts, ref_texts, meteor):
    """    
    Computes the METEOR score for a list of reference texts and corresponding predictions.

    Args:
        pred_texts: A list of predictions by the S2ST model.
        ref_texts: A list of target language sentences (reference translations).
        meteor: The huggingface instance that computes the METEOR metric.

    Returns:
        The vector of METEOR scores for the predictions.
    """

    return meteor.compute(predictions=[pred_texts], references=[[ref_texts]])['meteor']

def add_source_text_path_to_file(input_df, save_path = None):
    """
    Adds the source_text to a dataframe of references and predictions (both in target language),
    if it does not already contain it.
    
    Used before running the evaluation metrics in generate_metrics_for_file_vectorized(), since some of
    them require the source_text along with the pred_text and ref_text.

    Args:
        input_df: A DataFrame containing the pred_texts and ref_texts, but potentially lacking source_texts.
        save_path: Optional parameter to save the DataFrame with the source_text column included. Defaults to None.

    Returns:
        A DataFrame with the source_text column included
    """
    
    
    df = input_df
    if "source_text" in list(df.columns):
        return df
    
    else:
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

        i = 0
        for index, row in df.iterrows():
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
        
        if save_path:
            new_df.to_csv(save_path)
            
        return new_df

def generate_metrics_for_file_vectorized(filename, bleu, comet, meteor, blaser, text_embedder):
    """
    Generates the 4 translation metrics (ASR BLEU, COMET, METEOR, BLASER 2.0), along with
    BP (Brevity Penalty) and HRR (Hypothesis-Reference Ratio) for each sample in the input DataFrame.

    Args:
        filename: The path to the input DataFrame with at least pred_text and ref_text columns.
        bleu: The huggingface instance that computes the ASR-BLEU metric.
        comet: The huggingface instance that computes the COMET metric.
        meteor: The huggingface instance that computes the METEOR metric.
        blaser: The huggingface instance that computes the BLASER 2.0 metric.
        text_embedder: The text embedder model used by the blaser instance.

    Returns:
        The input DataFrame with 6 new columns (4 Translation metrics, BP, and HRR)
    """
    df = pd.read_csv(filename)
    df = df.replace(np.nan, '', regex=True)
    
    # Add Source Texts column if it doesn't already exist
    df = add_source_text_path_to_file(input_df=df)

    meteor_list, comet_list, blaser_list = [], [], []

    pred_texts = df["pred_text"]
    ref_texts = df["ref_text"]
    source_texts = df["source_text"]

    asr_bleu_list = [segment_bleu_score_string(str(bleu.sentence_score(text_normalizer(p), [text_normalizer(r)]) )  ) for p, r in zip(pred_texts, ref_texts)]
    
    bleu_score_list = [segment_bleu_score_string(s)[0] for s in asr_bleu_list]
    brevity_penalty_list = [segment_bleu_score_string(b)[1] for b in asr_bleu_list]
    hypothesis_reference_ratio_list = [segment_bleu_score_string(b)[2] for b in asr_bleu_list]

    
    comet_list = comet_score_vectorized(pred_texts=pred_texts, ref_texts=ref_texts, source_texts=source_texts, comet=comet)
    meteor_list = [meteor_score_vectorized(p, r, meteor) for p, r in zip(pred_texts, ref_texts)]
    blaser_list = blaser_score_vectorized(source_texts=source_texts, pred_texts=pred_texts, blaser=blaser, text_embedder=text_embedder)

    df["ASR_BLEU"] = bleu_score_list
    df["BP"] = brevity_penalty_list
    df["HRR"] = hypothesis_reference_ratio_list
    df["COMET"] = comet_list
    df["METEOR"] = meteor_list
    df["BLASER"] = blaser_list

    title = filename.split("/")[-1]
    df.to_csv(f"{OUTPUT_PATH}{title}")

    return df

def main():
    # Sample Usage
    blaser = load_blaser_model("blaser_2_0_qe").eval()
    text_embedder = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder")
    meteor = evaluate.load('meteor')
    comet = evaluate.load('comet')
    
    expanded_filename = "/home/aaditd/2_Speech_Project/metrics/raw/casc_finetuned_1e-7_1_epoch.csv"
    generate_metrics_for_file_vectorized(expanded_filename, comet, meteor, blaser, text_embedder)

    print("DONE!!")


if __name__ == "__main__":
    main()

