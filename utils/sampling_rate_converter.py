""" This module contains the function that is used to enforce a 16Khz
sampling rate for all files in the dev dataset.
"""

import os
from pydub import AudioSegment
from pydub.utils import mediainfo
from multiprocessing import Pool

def convert_file_speech_rate(input_file):
    
    converted_data_path = "/data/shire/data/aaditd/speech_data/source_dataset/clips_16"
    output_file = os.path.join(converted_data_path, os.path.basename(input_file))
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_frame_rate(16000)
    audio.export(output_file, format="mp3")
    
def main():
    folderA = "/data/shire/data/aaditd/speech_data/source_dataset/clips_medium"
    folderB = "/data/shire/data/aaditd/speech_data/source_dataset/clips_16"

    if not os.path.exists(folderB):
        os.makedirs(folderB)

    file_list = [os.path.join(folderA, f) for f in os.listdir(folderA) if f.endswith(".mp3")]

    with Pool() as pool:
        pool.map(convert_file_speech_rate, file_list)


if __name__ == "__main__":
    main()