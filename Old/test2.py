import os
import warnings
import sys

import librosa
import nltk
import numpy as np

import torch
from datasets import Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM

MODEL_ID = "bond005/wav2vec2-large-ru-golos-with-lm"
SAMPLES = 30

nltk.download('punkt')

processor = Wav2Vec2ProcessorWithLM.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    speech_array, _ = librosa.load(batch['path'], sr=16_000)
    batch['speech'] = np.asarray(speech_array, dtype=np.float32)
    return batch

def main():
    # Get files as commandline input from user
    filenames = sys.argv[1:]

    # Check that at least one file is provided and all files exist
    assert(len(filenames) > 0)
    assert(False not in [os.path.exists(file) for file in filenames])

    # Build a dataset from the files
    data = Dataset.from_dict({'path': filenames})

    # Convert audio to speech array
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = data.map(speech_file_to_array_fn)

    # Preprocess data to reduce noise and enhance
    inputs = processor(data['speech'], sampling_rate=16_000, return_tensors="pt", padding=True)
    
    # Get predicted transcription from model
    with torch.no_grad():
        logits = model(inputs.input_values,
                    attention_mask=inputs.attention_mask).logits
    
    transcriptions = processor.batch_decode(logits=logits.numpy()).text

    # Print results
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, predicted_sentence in enumerate(transcriptions):
            print("-" * 100)
            print("Audio file:", data['path'][i])
            print("Prediction:", predicted_sentence)

main()