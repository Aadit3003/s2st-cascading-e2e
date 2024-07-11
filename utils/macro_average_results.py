""" This module contains the function to macro average the Translation metrics
across all the samples in the dev dataset. 

It is used after running either of the inference modules 
(forward_feed_cascaded_finetuned.py or forward_feed_cascaded_oob.py)
"""

import pandas as pd
import os
import numpy as  np

def segment_bleu_score_string(bleu_score_string):
    """
    Extracts the float BLEU score, Brevity Penalty, and the  Hypothesis-Reference ratio
    values from the string returned by the huggingface instance that calculates the BLEU 
    score for a single translation. 
    
    Used in macro_average_translation_metrics() to obtain these three values from the BLEU
    score strings returned by the huggingface object.
    """
    parts = bleu_score_string.split(" ")
    main_score = float(parts[2])
    other_scores = [float(p) for p in parts[3].split("/")]
    bp = float(parts[6])
    ratio = float(parts[9])
        
    return (main_score, bp, ratio)
    
def macro_average_translation_metrics(s2st_model_inference_results_df):
    """
    Averages each of the 4 Translation metrics (BLEU, COMET, METEOR, BLASER 2.0),
    and the Brevity Penalty (BP), and Hypothesis-Reference Ratio (HRR) across all samples in
    the provided dataframe.
    
    To be used after calculating all the metrics using the expanded_translation_metric module.

    Args:
        s2st_model_inference_results_df: The dataframe containing each translation metric 
        (including BP and HRR) for every sample in the dev dataset. 

    Returns:
        The macro-averaged values of the translation metrics, BP, and HRR
    """
    

    comet_list = [float(a) for a in s2st_model_inference_results_df["COMET"]]
    meteor_list = [float(a) for a in s2st_model_inference_results_df["METEOR"]]
    blaser_list = [float(a) for a in s2st_model_inference_results_df["BLASER"]]
    bleu_score_list = [float(a) for a in s2st_model_inference_results_df["ASR_BLEU"]]
    brevity_penalty_list = [float(a) for a in s2st_model_inference_results_df["BP"]]
    hypothesis_reference_ratio_list = [float(a) for a in s2st_model_inference_results_df["HRR"]]

    avg_comet = round(np.sum(comet_list) / len(comet_list), 3)
    avg_meteor = round(np.sum(meteor_list) / len(meteor_list), 3)
    avg_blaser = round(np.sum(blaser_list) / len(blaser_list), 3)
    avg_bleu = round(np.sum(bleu_score_list) / len(bleu_score_list), 3)
    avg_bp = round(np.sum(brevity_penalty_list) / len(brevity_penalty_list), 3)
    avg_ratio = round(np.sum(hypothesis_reference_ratio_list) / len(hypothesis_reference_ratio_list), 3)


    return avg_bleu, avg_bp, avg_ratio, avg_comet, avg_meteor, avg_blaser

def main():
    
    METRICS_DIRECTORY = "./metrics/scored/"

    for file in sorted(os.listdir(METRICS_DIRECTORY)):
        df = pd.read_csv(METRICS_DIRECTORY+file)
        avg_asr, avg_bp, avg_ratio, avg_comet, avg_meteor, avg_blaser = macro_average_translation_metrics(df)
        print(f"{file} | ASR_BLEU = {avg_asr} | BP = {avg_bp} | HRR = {avg_ratio} | COMET = {avg_comet} | METEOR = {avg_meteor} | BLASER = {avg_blaser}")


if __name__ == "__main__":
    main()
    