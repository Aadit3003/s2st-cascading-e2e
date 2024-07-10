""" This module contains the function to macro average the Translation metrics
across all the samples in the dev dataset. 

It is used after running either of the inference modules 
(forward_feed_cascaded_finetuned.py or forward_feed_cascaded_oob.py)
"""

import pandas as pd
import os
import numpy as  np

def segment_bleu_score_string(bleu_score_string):
        parts = bleu_score_string.split(" ")
        main_score = float(parts[2])
        other_scores = [float(p) for p in parts[3].split("/")]
        bp = float(parts[6])
        ratio = float(parts[9])
        
        return (main_score, bp, ratio)
    
def get_ASR_BLEU(s2st_model_inference_results_df):
    
    asr_bleu_score_string_list = s2st_model_inference_results_df["ASR_BLEU"]

    comet_list = [float(a) for a in s2st_model_inference_results_df["COMET"]]
    meteor_list = [float(a) for a in s2st_model_inference_results_df["METEOR"]]
    blaser_list = [float(a) for a in s2st_model_inference_results_df["BLASER"]]



    bleu_score_list = [segment_bleu_score_string(s)[0] for s in asr_bleu_score_string_list]
    brevity_penalty_list = [segment_bleu_score_string(b)[1] for b in asr_bleu_score_string_list]
    hypothesis_reference_ratio_list = [segment_bleu_score_string(b)[2] for b in asr_bleu_score_string_list]



    avg_bleu = round(np.sum(bleu_score_list) / len(bleu_score_list), 3)
    avg_bp = round(np.sum(brevity_penalty_list) / len(brevity_penalty_list), 3)
    avg_ratio = round(np.sum(hypothesis_reference_ratio_list) / len(hypothesis_reference_ratio_list), 3)

    avg_comet = round(np.sum(comet_list) / len(comet_list), 3)
    avg_meteor = round(np.sum(meteor_list) / len(meteor_list), 3)
    avg_blaser = round(np.sum(blaser_list) / len(blaser_list), 3)

    return avg_bleu, avg_bp, avg_ratio, avg_comet, avg_meteor, avg_blaser

def main():
    
    METRICS_DIRECTORY = "./metrics/scored/"

    for file in sorted(os.listdir(METRICS_DIRECTORY)):
        df = pd.read_csv(METRICS_DIRECTORY+file)
        avg_asr, avg_bp, avg_ratio, avg_comet, avg_meteor, avg_blaser = get_ASR_BLEU(df)
        print(f"{file} | ASR_BLEU = {avg_asr} | BP = {avg_bp} | HRR = {avg_ratio} | COMET = {avg_comet} | METEOR = {avg_meteor} | BLASER = {avg_blaser}")


if __name__ == "__main__":
    main()
    