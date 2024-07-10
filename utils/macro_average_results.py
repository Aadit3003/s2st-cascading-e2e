""" This module contains the function to macro average the Translation metrics
across all the samples in the dev dataset. 

It is used after running either of the inference modules 
(forward_feed_cascaded_finetuned.py or forward_feed_cascaded_oob.py)
"""

import pandas as pd
import os
import numpy as  np

# PREFIX_1 = "/home/aaditd/2_Speech_Project/results/e2e_results/"
# PREFIX_2 = "/home/aaditd/2_Speech_Project/results/casc_results/"
# RESULTS_PATH_1 = "/home/aaditd/2_Speech_Project/results/e2e_results/e2e_output_final.csv"
# RESULTS_PATH_2 = "/home/aaditd/2_Speech_Project/results/casc_results/casc_output_final.csv"
# PATH = "/home/aaditd/2_Speech_Project/frog.csv"

# df1 = pd.read_csv(RESULTS_PATH_1)
# df2 = pd.read_csv(RESULTS_PATH_2)
# df = pd.read_csv(PATH)

def get_ASR_BLEU(df):
    asr_long_list = df["ASR_BLEU"]

    comet_list = [float(a) for a in df["COMET"]]
    meteor_list = [float(a) for a in df["METEOR"]]
    blaser_list = [float(a) for a in df["BLASER"]]

    def get_stuff(string):
        parts = string.split(" ")
        main_score = float(parts[2])
        other_scores = [float(p) for p in parts[3].split("/")]
        bp = float(parts[6])
        ratio = float(parts[9])
        # print(main_score, bp)
        return (main_score, bp, ratio)
        return parts, (main_score, other_scores, bp, ratio)

    # a, b = get_stuff(asr_long_list[0])
    # print(a)
    # print(b)


    MAIN_LIST = [get_stuff(s)[0] for s in asr_long_list]
    BP_LIST = [get_stuff(b)[1] for b in asr_long_list]
    R_LIST = [get_stuff(b)[2] for b in asr_long_list]
    # print(BP_LIST)


    avg_asr = round(np.sum(MAIN_LIST) / len(MAIN_LIST), 3)
    avg_bp = round(np.sum(BP_LIST) / len(BP_LIST), 3)
    avg_ratio = round(np.sum(R_LIST) / len(R_LIST), 3)

    avg_comet = round(np.sum(comet_list) / len(comet_list), 3)
    avg_meteor = round(np.sum(meteor_list) / len(meteor_list), 3)
    avg_blaser = round(np.sum(blaser_list) / len(blaser_list), 3)

    return avg_asr, avg_bp, avg_ratio, avg_comet, avg_meteor, avg_blaser


FOLDER = "/home/aaditd/2_Speech_Project/metrics/scored/"


for file in sorted(os.listdir(FOLDER)):
    df = pd.read_csv(FOLDER+file)
    avg_asr, avg_bp, avg_ratio, avg_comet, avg_meteor, avg_blaser = get_ASR_BLEU(df)
    print(f"{file} | ASR_BLEU = {avg_asr} | BP = {avg_bp} | HRR = {avg_ratio} | COMET = {avg_comet} | METEOR = {avg_meteor} | BLASER = {avg_blaser}")

