import sys
import os
import numpy as np
import torch
import math
from datasets import Dataset
from faster_whisper import WhisperModel

# Using the ported transformers version of the fairseq translation model for
# a lighter weight
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

DIR_PATH = '/Users/shravya/Projects/speech.ru/translate/'

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
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return formatted_time

def generate_subtitle_file(filename, language, segments):
    tmp = filename.split('/')
    file = tmp[-1].split('.')[0]
    sub_file = f'sub-{file}.{language}.vtt'
    subtitle_file = os.path.join(DIR_PATH, 'public', sub_file)
    text = 'WEBVTT\n\n'
    for index, segment in enumerate(segments):
        segment_start = format_time(segment['start'])
        segment_end = format_time(segment['end'])
        text += f'{str(index+1)} \n'
        text += f'{segment_start} --> {segment_end} \n'
        text += f'{segment["text"]} \n'
        text += '\n'
        
    f = open(subtitle_file, 'w')
    f.write(text)
    f.close()

    return sub_file

def run(filenames):
    data = Dataset.from_dict({'path': filenames})

    results = {}

    with torch.no_grad():
        for d in data:
            filename = d['path']
            results[filename] = {'ru_subs': None, 'en_subs': None, 'align': None}

            segments, _ = transcript_model.transcribe(filename,
                                                         language='ru')
            transcribed_segments = list(map(lambda s: {'start': s.start,
                                                       'end': s.end,
                                                       'text': s.text},
                                            segments))

            # Generate Russian subtitle track
            results[filename]['ru_subs'] = generate_subtitle_file(filename, 'ru',
                                                                  transcribed_segments)

            # Encode all the transcripted phrases
            encoded_phrases = list(map(lambda x: tokenizer.encode(x['text'],
                                                                  return_tensors='pt'),
                                       transcribed_segments))
                        
            translated_segments = []
            alignment = []

            for i, phrase in enumerate(encoded_phrases):
                generated = ru2en.generate(phrase,
                                           output_attentions=True,
                                           return_dict_in_generate=True)

                # Batch size is 1 so just grab the first element of output tokens
                output = generated.sequences[0]
                result = tokenizer.decode(output, skip_special_tokens=True)

                # Update segment with translation
                start = transcribed_segments[i]['start']
                end = transcribed_segments[i]['end']
                translated_segments.append({'start': start, 'end': end, 'text': result})

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
                    
            # Generate English subtitle track
            results[filename]['en_subs'] = generate_subtitle_file(filename, 'en',
                                                                  translated_segments)

            results[filename]['align'] = alignment

    return results

if __name__ == '__main__':
    results = None
    if len(sys.argv) < 2:
        results = run(None)
    else:
        results = run(sys.argv[1:])

    for k,v in results.items():
        print(f'--------------- {k} ---------------')
        print(f'ru_subs: {v["ru_subs"]}')
        print(f'en_subs: {v["en_subs"]}')
        print(f'alignment: {v["align"]}')
        print()