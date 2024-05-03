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

            translated_segments = []
            alignment = []

            for i, segment in enumerate(transcribed_segments):
                text = segment['text']
                ids = []
                input_to_ids = []
                count = 0
                # Encode text and prepare to give as input to the model
                for i, word in enumerate(text.split()):
                    encoded = tokenizer.encode_plus(word,
                                                    add_special_tokens=False,
                                                    return_tensors='pt')
                    input_ids = list(encoded['input_ids'])
                    ids += input_ids
                    input_to_ids.append((count, count + len(ids)))
                    print(i, word)

                # Add in special tokens for model input
                input = tf.constant([tokenizer.build_inputs_with_special_tokens(ids)])

                align = []
                generated = ru2en.generate(input,
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
                for i in range(len(output)):
                    # Attention information for the current token
                    attention_info = generated.cross_attentions[i]
                    attn = []
                    # Next is attention heads
                    for attn_head in attention_info:
                        # Next is number of beams, want the top one
                        beam = attn_head[0]
                        # Next is the number of layers, want the last one
                        last_layer = beam[-1]
                        # Next is the batch size, always 1
                        attn.append(last_layer[0])

                    attn = np.asarray(attn).mean(axis=0)
                    # Want to get value from across attention heads, without
                    # considering special tokens added by the tokenizer
                    special_tokens_mask = tokenizer.get_special_tokens_mask(
                        input_ids[i], already_has_special_tokens=True)
                    special_tokens_indices = [j for j, v in enumerate(special_tokens_mask) if v == 1]
                    attn.delete([special_tokens_indices], 1)

                    max_index, max_value = None
                    for i, (start, end) in enumerate(input_to_ids):
                        mi = attn[start:end].argmax().item()
                        value = attn[mi]
                        if value > max_value:
                            max_index = i
                            max_value = value
                    
                    if max_index:
                        align.append((max_index, i))
                
                alignment.append(align)
                print(alignment)

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