# python code to convert wav fils from 44.1kHz to 16kHz
# arguments: input_dir, output_dir
#
import os
import subprocess

# Define the source directory containing the audio files
source_dir = '/data/EmoFilm/wav_corpus'
target_dir = '/data/EmoFilm/wav_corpus_16k'

# create the target directory if it does not exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Define the target sample rate
target_sr = 16000

# Loop over all audio files in the source directory
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith('.wav'):
            # Define the input and output file paths
            input_path = os.path.join(root, file)
            # obtain the basename
            basename = os.path.basename(input_path)
            output_path = os.path.join(target_dir, basename[:-4] + '_16k.wav')
           
            print(f"Resampling {basename} to {output_path}")
            # Use sox to resample the audio file
            subprocess.run(['sox', input_path, '-r', str(target_sr), output_path])