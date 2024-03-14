#!/usr/bin/env python
# coding: utf-8
# 31 Jan 2024
# dataloader for CV data with accent

from datasets import DatasetDict
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2Processor
import torch, pickle, random,json,re, io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataclasses import dataclass
import numpy as np

device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


SAMPLERATE=16000

repl_sep_space = lambda s:re.sub('\|+', ' ', s)
repl_sep_line = lambda s:re.sub(' ', '|', s)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


class stackup_inputs(object):
    def __init__(self):
        pass
        
    def __call__(self, sample):
        utt = sample['utt']
        txt = sample['txt']
        lbl = torch.tensor(sample['lbl'])
        act = sample['act']
        
        return utt, lbl, txt, act

class Dataloader_utts_cv(Dataset):
    def __init__(self,  dataset_file='', vocabjson='', 
                 transform=None, category='train', refmodeldir=''): # category one of ['train','val','test']
        
        self.transform=transform
        self.model = Wav2Vec2ForCTC.from_pretrained(refmodeldir)
        dset=DatasetDict()
        self.dset=dset.load_from_disk(dataset_file)[category]
        self.accentlist=list(set(self.dset['accent']))
        self.num_rows=self.dset.num_rows
        
        with open(vocabjson,'r') as f:
            self.charmap=json.load(f)
        self.onlychars=[ch for ch in self.charmap.keys() if ch not in ['[PAD]','[UNK]']]
        #self.onlychars=['_']+[ch if ch !='|' else 'sil' for ch in self.onlychars]
        
        self.onlychars.sort()
        self.onlychars=['_']+self.onlychars
        self.ctc_charmap={}
        for i,ch in enumerate(self.onlychars):
            self.ctc_charmap[i]=ch
            self.ctc_charmap[ch]=i
            
        #self.model = accelerator.prepare(self.model)
        self.model = self.model.to(device)
            
    def __len__(self):
        return self.num_rows
    
    def __getitem__(self, ix):
        utt=self.dset[ix]['audio']['array']
        waveform=torch.from_numpy(utt.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            
            #features
            fout=self.model.wav2vec2.feature_extractor(waveform.to(device))
            #projection
            fpout=self.model.wav2vec2.feature_projection(fout.permute(0,2,1))[0]
            #encoder out
            eout=self.model.wav2vec2.encoder(fpout).last_hidden_state
            
        txt=self.dset[ix]['text']
        txt=repl_sep_line(txt)
        lbl=[self.ctc_charmap[ch] for ch in txt]
        act=self.dset[ix]['accent']

        sample={'utt':eout[0],'txt':txt, 'lbl':lbl, 'act':act}
        if self.transform:
            sample = self.transform(sample)
        return sample

def collate_wrapper(batch):
    utts=[];lbls=[];txts=[];acts=[]
    for i,item in enumerate(batch):
        utt, lbl, txt, act = item
        utts.append(utt)
        lbls.append(lbl)
        txts.append(txt)
        acts.append(act)
        
    return utts, lbls, txts, acts

