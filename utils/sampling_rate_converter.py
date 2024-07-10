""" This module contains the function that is used to enforce a 16Khz
sampling rate for all files in the dev source dataset.
"""

import os

from pydub import AudioSegment
from pydub.utils import mediainfo
from multiprocessing import Pool

def process_file(input_file):
    output_file = os.path.join(folderB, os.path.basename(input_file))
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_frame_rate(16000)
    audio.export(output_file, format="mp3")

if __name__ == "__main__":
    folderA = "/data/shire/data/aaditd/speech_data/source_dataset/clips_medium"
    folderB = "/data/shire/data/aaditd/speech_data/source_dataset/clips_16"

    if not os.path.exists(folderB):
        os.makedirs(folderB)

    file_list = [os.path.join(folderA, f) for f in os.listdir(folderA) if f.endswith(".mp3")]

    with Pool() as pool:
        pool.map(process_file, file_list)

