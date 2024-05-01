import sys
import warnings
import numpy as np
import torch
import math
from datasets import Dataset
from faster_whisper import WhisperModel

# Using the ported transformers version of the fairseq translation model for
# a lighter weight
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

# Transcription model
transcript_model = WhisperModel('models/large')

# Russian to English translation model
TRANSLATION_MODEL_ID = 'facebook/wmt19-ru-en'
tokenizer = FSMTTokenizer.from_pretrained(TRANSLATION_MODEL_ID)
en2ru_tokenizer = FSMTTokenizer.from_pretrained('facebook/wmt19-en-ru')
ru2en = FSMTForConditionalGeneration.from_pretrained(TRANSLATION_MODEL_ID)
ru2en.eval()

# Return list of indices of special tokens added by the tokenizer.
def get_special_indices(tokens):
    mask = tokenizer.get_special_tokens_mask(tokens, already_has_special_tokens=True)
    return [i for i, v in enumerate(mask) if v == 1]

def format_time(seconds):
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:01d},{milliseconds:03d}"
    return formatted_time

def generate_subtitle_file(filename, language, segments):
    base_filename = filename.split('.')[0]
    subtitle_file = f'sub-{base_filename}.{language}.srt'
    text = ''
    for index, segment in enumerate(segments):
        segment_start = format_time(segment.start)
        segment_end = format_time(segment.end)
        text += f'{str(index+1)} \n'
        text += f'{segment_start} --> {segment_end} \n'
        text += f'{segment.text} \n'
        text += '\n'
        
    f = open(subtitle_file, 'w')
    f.write(text)
    f.close()

    return subtitle_file

def run(filenames):
    data = Dataset.from_dict({'path': filenames})

    transcriptions = []
    translations = []
    alignments = []

    with torch.no_grad():
        for d in data:
            segments, _ = transcript_model.transcribe(d['path'],
                                                         language='ru')
            segments = list(segments)
            for segment in segments:
                # Print segment information
                print('[%.2fs -> %.2fs] %s' %
                    (segment.start, segment.end, segment.text))
                
            transcript_chunks = list(map(lambda x: x.text, segments))
                    
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
                    
                    max_index = attention.argmax().item()
                    input_word = en2ru_tokenizer.decode([phrase[0, max_index]], skip_special_tokens=True)
                    output_word = tokenizer.decode([token], skip_special_tokens=True)
                    alignment.append((input_word, output_word))
                    
            translations.append(' '.join(translated))
            alignments.append(alignment)
    
    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i, predicted_sentence in enumerate(transcriptions):
            fname = filenames[i] if filenames else f'test_{i}'
            results[fname] = {'transcription': predicted_sentence,
                              'translation': translations[i],
                              'alignment': alignments[i]}
    return results

if __name__ == '__main__':
    results = None
    if len(sys.argv) < 2:
        results = run(None)
    else:
        results = run(sys.argv[1:])

    for k,v in results.items():
        print(f'--------------- {k} ---------------')
        print(f'Transcription: {v["transcription"]}')
        print(f'Translation: {v["translation"]}')
        print(f'Alignment: {v["alignment"]}')
        print()