import warnings
import io
import logging

import librosa
import soundfile as sf
import nltk
import numpy as np

import torch
from datasets import Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM

# Using the ported transformers version of the fairseq translation model for
# a lighter weight
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

# Set the logging level to WARNING to suppress INFO and DEBUG messages
logging.basicConfig(level=logging.WARNING)

TRANSCRIPT_MODEL_ID = "bond005/wav2vec2-large-ru-golos-with-lm"
DATASET_ID = "bond005/sberdevices_golos_10h_crowd"
SAMPLES = 1

nltk.download('punkt', quiet=True)

processor = Wav2Vec2ProcessorWithLM.from_pretrained(TRANSCRIPT_MODEL_ID)
transcript_model = Wav2Vec2ForCTC.from_pretrained(TRANSCRIPT_MODEL_ID)

# Use the Russian to English translation model
TRANSLATION_MODEL_ID = "facebook/wmt19-ru-en"
tokenizer = FSMTTokenizer.from_pretrained(TRANSLATION_MODEL_ID)
en2ru_tokenizer = FSMTTokenizer.from_pretrained("facebook/wmt19-en-ru")
ru2en = FSMTForConditionalGeneration.from_pretrained(TRANSLATION_MODEL_ID)
ru2en.eval()

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_inputs_fn(batch):
    data, _ = sf.read(batch['file'])
    speech_array, sr = librosa.resample(data, sr=16_000)
    
    # Split audio file into smaller chunks so model can process all data
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

# Return list of indices of special tokens added by the tokenizer.
def get_special_indices(tokens):
    mask = tokenizer.get_special_tokens_mask(tokens, already_has_special_tokens=True)
    return [i for i, v in enumerate(mask) if v == 1]

def run(file):
    # Build a dataset from the file
    data = Dataset.from_dict({'file': io.BytesIO(file.read())})
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
                    
            # Link together the transcribed chunks
            transcriptions.append(' '.join(transcript_chunks))

            # Encode all the transcripted phrases
            encoded_phrases = list(map(lambda x: tokenizer.encode(x, return_tensors="pt"),
                                       transcript_chunks))
            
            translated = []
            alignment = []

            for phrase in encoded_phrases:
                generated = ru2en.generate(phrase,
                                           output_attentions=True,
                                           return_dict_in_generate=True)

                # Batch size is 1 so just grab the first element of output tokens
                output = generated.sequences[0]
                result = tokenizer.decode(output, skip_special_tokens=True)
                translated.append(result)
                # For each output token
                for i, token in enumerate(output):
                    # Don't do this for special tokens
                    if i in get_special_indices(output):
                        continue

                    # Attention information for the current token
                    attention_info = generated.cross_attentions[i]
                    attention = []
                    # Next is attention heads
                    for attn_head in attention_info:
                        # Next is number of beams, want the top one
                        beam = attn_head[0]
                        # Next is the number of layers, want the last one
                        last_layer = beam[-1]
                        # Next is the batch size, always 1
                        attention.append(last_layer[0])

                    attention = np.asarray(attention).mean(axis=0)
                    # Want to get value from across attention heads, without
                    # considering special tokens added by the tokenizer
                    special_tokens_mask = tokenizer.get_special_tokens_mask(
                        phrase[0], already_has_special_tokens=True)
                    special_tokens_indices = [i for i, v in enumerate(special_tokens_mask) if v == 1]
                    attention[special_tokens_indices] = 0
                    
                    print(attention)
                    max_index = attention.argmax().item()
                    input_word = en2ru_tokenizer.decode([phrase[0, max_index]], skip_special_tokens=True)
                    output_word = tokenizer.decode([token], skip_special_tokens=True)
                    alignment.append((input_word, output_word))
                    
            translations.append(' '.join(translated))
            alignments.append(alignment)
    
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, predicted_sentence in enumerate(transcriptions):
            results.append({"prediction": predicted_sentence,
                            "translation": translations[i],
                            "alignment": alignments[i]})
    return results