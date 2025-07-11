
### Inference whisper model original with various corpus
### check using selected set of test datasets
### https://huggingface.co/learn/audio-course/en/chapter5/evaluation

from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline
import torch
from transformers import WhisperForConditionalGeneration
from transformers import WhisperTokenizer
from transformers import WhisperFeatureExtractor
from transformers import WhisperProcessor
from datasets import DatasetDict
import json
import numpy as np

import evaluate

seed=25
torch.manual_seed(seed)
np.random.seed(seed)

metric = evaluate.load("cer")

if torch.cuda.is_available():
    device = "cuda:0"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32

msg= 'all' #'indic' #'all' # 'others'

def get_cer(tdset):
    preds=[]
    for pred in pipe(KeyDataset(tdset, "audio")):
        preds.append(pred['text'])
    cer=metric.compute(predictions=preds, references=tdset["text"])
    return cer

results={}

whispermodels=['whisper-tiny','whisper-base', 'whisper-small', 'whisper-medium', 'whisper-large-v3']
for whispermodel in whispermodels:
    modeldesc=f'openai/{whispermodel}'

    tokenizer = WhisperTokenizer.from_pretrained(modeldesc, language="English", task="transcribe")
    processor = WhisperProcessor.from_pretrained(modeldesc, language="English", task="transcribe")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/{whispermodel}")

    model = WhisperForConditionalGeneration.from_pretrained(f"openai/{whispermodel}")

    model.generation_config.language = "english"

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        chunk_length_s=30,
        torch_dtype=torch_dtype,
        device=device,
    )

    print(f'Inference using {whispermodel}...\n')

    results[whispermodel]={}

    if msg in ['indic','all']:
        datadir="datasets/IndicTimit"
        langs=["HIN", "TAM", "BEN", "MLY", "MAR", "KAN"]
        for lang in langs:
            print(f'--> inference on IndicTimit-{lang}')
            dataset_name=f'{datadir}/{lang}' if lang=='HIN' else f'{datadir}/{lang}_test'
            dset=DatasetDict().load_from_disk(dataset_name)
            test_dset=dset['test']
            cer=get_cer(test_dset)
            results[whispermodel][f'IndicTimit-{lang}']=f'{cer*100:.2f}%'
            
            
        print(f'Inference on {whispermodel}-{msg}: cer% - all utts...')
        print(results)

    if msg in ['others','all']:
        others=["CV1group","CV1_gen","svarah","timit","LibriTest"]
        dsetnames=[["CV1_uki","CV1_oriental","CV1_northam","CV1_african","CV1_anz"],[],[],[], ["libricleaned"]]
        
        for i, other in enumerate(others):
            dsetname=dsetnames[i]
            if other=="CV1group":
                dset_cat="train"
            else:
                dset_cat="test"
            if dsetname==[]:
                print(f'--> inference on {other}')
                dsetdir=f'datasets/{other}'
                dset=DatasetDict().load_from_disk(dsetdir)
                # print(f'Processing {other}-{dset_cat} {dset[dset_cat].num_rows} utts...')
                test_dset=dset[dset_cat]
                cer=get_cer(test_dset)
                results[whispermodel][other]=f'{cer*100:.2f}%'
            else:
                for sub in dsetname:
                    print(f'--> inference on {other}-{sub}')
                    dsetdir=f'datasets/{other}/{sub}'
                    dset=DatasetDict().load_from_disk(dsetdir)
                    # print(f'Processing {other}-{sub}-{dset_cat} {dset[dset_cat].num_rows} utts...')
                    test_dset=dset[dset_cat]
                    cer=get_cer(test_dset)
                    results[whispermodel][sub]=f'{cer*100:.2f}%'
            

    print(f'Inference on {whispermodel} - {msg}: cer% - all utts...')
    print(results[whispermodel])
    print('--------------------------------------')
    
    fnt=f'inference_zeroshot_{whispermodel}_{msg}.json'
    with open(fnt,'w') as f:
        json.dump(results,f,indent=4)

    del model, tokenizer, processor, feature_extractor

