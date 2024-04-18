import os
import warnings
import sys
import logging

import librosa
import nltk
import numpy as np

import torch
from datasets import Dataset, load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM

# Set the logging level to WARNING to suppress INFO and DEBUG messages
logging.basicConfig(level=logging.WARNING)

TRANSCRIPT_MODEL_ID = "bond005/wav2vec2-large-ru-golos-with-lm"
DATASET_ID = "bond005/sberdevices_golos_10h_crowd"
SAMPLES = 5

nltk.download('punkt', quiet=True)

processor = Wav2Vec2ProcessorWithLM.from_pretrained(TRANSCRIPT_MODEL_ID)
transcript_model = Wav2Vec2ForCTC.from_pretrained(TRANSCRIPT_MODEL_ID)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en',
                        checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                        tokenizer='moses', bpe='fastbpe')
    # Disable dropout
    ru2en.eval()

# TRANSLATE_MODEL_ID = "facebook/nllb-200-distilled-600M"
# tokenizer = AutoTokenizer.from_pretrained(TRANSLATE_MODEL_ID)
# translate_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATE_MODEL_ID)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_inputs_fn(batch):
    speech_array, sr = librosa.load(batch['path'], sr=16_000)
    
    # Split audio file into one minute chunks so model can process all data
    sample_len = 30 * sr
    counter = 0
    samples_array = []
    while counter < len(speech_array):
        sample_amt = min(sample_len, len(speech_array) - counter)
        sample = speech_array[counter : counter + sample_amt]
        samples_array.append(np.asarray(sample, dtype=np.float32))
        counter += sample_amt
    batch['speech'] = samples_array
    return batch

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    speech_array = batch["audio"]["array"]
    batch["speech"] = [np.asarray(speech_array, dtype=np.float32)]
    return batch

def main():
    testMode = False

    # Get files as commandline input from user
    filenames = sys.argv[1:]

    # Check that at least one file is provided and all files exist
    if len(filenames) == 0:
        testMode = True
    else:
        assert(False not in [os.path.exists(f) for f in filenames])

    # Build a dataset from the files
    data = None
    if testMode:
        data = load_dataset(DATASET_ID, split=f"test[:{SAMPLES}]")
        removed_columns = set(data.column_names)
        removed_columns -= {'transcription', 'speech'}
        removed_columns = sorted(list(removed_columns))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = data.map(
                speech_file_to_array_fn,
                remove_columns=removed_columns
            )
    else:
        data = Dataset.from_dict({'path': filenames})
        # Convert audio to speech array
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = data.map(speech_file_to_inputs_fn)

    transcriptions = []
    translations = []
    alignments = []

    # Get predicted transcription from model
    with torch.no_grad():
        for d in data:
            transcript_chunks = []
            # Iterate over all audio chunks for each file
            for row in d['speech']:
                # Transcribe the audio first
                inputs = processor(row,
                                sampling_rate=16_000,
                                return_tensors="pt",
                                padding=True)
                logits = transcript_model(inputs.input_values,
                                        attention_mask=inputs.attention_mask).logits
                tscript = processor.decode(logits=logits.numpy()[:][:][0]).text
                transcript_chunks.append(tscript)

                # To translate, first encode, then run model, then decode
                # translation = []
                # for t in transcription:
                    # input_ids = tokenizer.encode(t, return_tensors="pt")
                    # output_ids = translate_model.generate(input_ids, max_new_tokens=100)
                    # translation.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))
                    
            # Link together the transcribed chunks
            transcriptions.append(' '.join(transcript_chunks))

            # Encode all the translated phrases
            encoded_phrases = list(map(lambda x: ru2en.encode(x), transcript_chunks))
            
            translated = []
            alignment = []
            for results in ru2en.generate(encoded_phrases):
                # Pick top scoring translation
                best_trans = results[0]
                translated.append(ru2en.decode(best_trans['tokens']))
                
                # Attention matrix has rows of output tokens corresponding to attention
                # weights for each source token (columns)
                # So to find which input token affected the output token most, find
                # argmax across rows
                alignment.append(best_trans['attention'].argmax(dim=1))
                    
            translations.append(' '.join(translated))
            alignments.append(alignment)
    
    # Print results
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, predicted_sentence in enumerate(transcriptions):
            print("-" * 100)
            if testMode:
                print("Reference:", data['transcription'][i])
            else:
                print("Audio file:", data['path'][i])
            print("Prediction:", predicted_sentence)
            print("Translation:", translations[i])
            print("Alignments:", alignments[i])

main()