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

DIR_PATH = './translate/'

# Transcription model
transcript_model = WhisperModel('large-v3')

COLORS = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0',
          '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',
          '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff']

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

            # Encode all the transcripted phrases and create a mapping to link
            # tokens to input words
            encoded_phrases = list(map(lambda x: tokenizer.encode(x['text'],
                                                                  return_tensors='pt'),
                                       transcribed_segments))
                        
            translated_segments = []
            alignment = []

            for i, phrase in enumerate(encoded_phrases):
                input_words = []
                tokens = []
                count = 0
                for tok in en2ru_tokenizer.convert_ids_to_tokens(phrase[0].tolist()):
                    if tok == '</s>':
                        count += 1
                        continue
                    toks = tok.split('</w>')
                    tokens.append(toks[0])
                    if len(toks) > 1:
                        input_words.append((
                            tokenizer.convert_tokens_to_string(tokens),
                            (count, count + len(tokens))))
                        count += len(tokens)
                        tokens = []

                generated = ru2en.generate(phrase,
                                           output_attentions=True,
                                           return_dict_in_generate=True)

                # Batch size is 1 so just grab the first element of output tokens
                output = generated.sequences[0]
                output_words = []
                tokens = []
                count = 0
                for tok in tokenizer.convert_ids_to_tokens(output.tolist()):
                    if tok == '</s>':
                        count += 1
                        continue
                    toks = tok.split('</w>')
                    tokens.append(toks[0])
                    if len(toks) > 1:
                        output_words.append((
                            en2ru_tokenizer.convert_tokens_to_string(tokens),
                            (count, count + len(tokens))))
                        count += len(tokens)
                        tokens = []

                # At this point, input_words maps a full input word to indices
                # and output_words maps a full output word to token indices
                
                result = tokenizer.decode(output, skip_special_tokens=True)

                # Update segment with translation
                start = transcribed_segments[i]['start']
                end = transcribed_segments[i]['end']
                translated_segments.append({'start': start, 'end': end, 'text': result})
                
                input_colors = [COLORS[c % len(COLORS)] for c in range(len(input_words))]
                output_colors = [[] for _ in range(len(output_words))]
                # For each output token
                for output_index, (_, (start, end)) in enumerate(output_words):
                    # Attention information for the current tokens
                    attention_info = generated.cross_attentions[start:end]
                    
                    attns = []
                    for token_info in attention_info:
                        attn = []
                        # Next is attention heads
                        for attn_head in token_info:
                            # Next is number of beams, want the top one
                            beam = attn_head[0]
                            # Next is the number of layers, want the last one
                            last_layer = beam[-1]
                            # Next is the batch size, always 1
                            attn.append(last_layer[0])
                        # Average across attention heads
                        attns.append(np.asarray(attn).mean(axis=0))

                    # Max attention score across tokens
                    attention = np.asarray(attns).max(axis=0)

                    # Want to get value from across attention heads, without
                    # considering special tokens added by the tokenizer
                    special_tokens_mask = tokenizer.get_special_tokens_mask(
                        phrase[0], already_has_special_tokens=True)
                    special_tokens_indices = [j for j, v in enumerate(special_tokens_mask) if v == 1]
                    attention[special_tokens_indices] = 0

                    max_index = attention.argmax().item()

                    # Using max_index, find the input word that matches
                    for input_index, (_, (start, end)) in enumerate(input_words):
                        if max_index in range(start, end):
                            output_colors[output_index] = input_colors[input_index]
                            break

                alignment.append(((translated_segments[i]['start'],
                                   translated_segments[i]['end']),
                                   (list(map(lambda x: x[0], input_words)), input_colors),
                                   (list(map(lambda x: x[0], output_words)), output_colors)))
            
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