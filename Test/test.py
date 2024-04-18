import os
import warnings

import librosa
import nltk
import numpy as np

import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM

MODEL_ID = "bond005/wav2vec2-large-ru-golos-with-lm"
DATASET_ID = "bond005/sberdevices_golos_10h_crowd"
SAMPLES = 5

nltk.download('punkt')
num_processes = max(1, os.cpu_count())

test_dataset = load_dataset(DATASET_ID, split=f"test[:{SAMPLES}]")
processor = Wav2Vec2ProcessorWithLM.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    speech_array = batch["audio"]["array"]
    batch["speech"] = np.asarray(speech_array, dtype=np.float32)
    return batch

removed_columns = set(test_dataset.column_names)
removed_columns -= {'transcription', 'speech'}
removed_columns = sorted(list(removed_columns))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    test_dataset = test_dataset.map(
        speech_file_to_array_fn,
        num_proc=num_processes,
        remove_columns=removed_columns
    )

inputs = processor(test_dataset["speech"], sampling_rate=16_000,
                   return_tensors="pt", padding=True)
with torch.no_grad():
    logits = model(inputs.input_values,
                   attention_mask=inputs.attention_mask).logits
predicted_sentences = processor.batch_decode(
    logits=logits.numpy(),
    num_processes=num_processes
).text

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for i, predicted_sentence in enumerate(predicted_sentences):
        print("-" * 100)
        print("Reference:", test_dataset[i]["transcription"])
        print("Prediction:", predicted_sentence)
