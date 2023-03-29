import yaml
import argparse
import json
import random
import os
import torch
import tqdm
import torch.utils.data as Data
import yaml
import logging
from collections import Counter
from sklearn.metrics import f1_score
import spacy
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset

def GetLogger(testName):
    logger = logging.getLogger(testName)
    logger.setLevel(logging.INFO)
    hdlr = logging.StreamHandler()
    hdlr.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    return logger
logger = GetLogger('data')

def read_json(path):
    # Manually open because .splitlines is different from iterating over lines
    with open(path, "r") as f:
        lines = json.load(f)
    for line in lines:
        yield line

def build_dataset(dataset):
    sentences = []
    labels = []

    for sample in open('/users12/rzliu/workspace/chatglm335M_zh/Belle.train.json'):
        sample = json.loads(sample)
        sentence = sample['input'].replace('\n', '').replace('\\', '')
        label = sample['target']
        sentence += ' [gMASK] <|startofpiece|>'
        sentences.append(sentence)
        labels.append(label)
        if len(sentences) > 52000:
            break
    
    return DatasetIterater(sentences, labels)





class DatasetIterater(Data.Dataset):    
    """自定义数据集类只需要重写__len__和__getitem__方法"""
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained("/users12/rzliu/workspace/chatglm335M_zh/chineseTokenizer", trust_remote_code=True)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        results = self.tokenizer(sentence,
                        padding="max_length",
                        truncation=True,
                        max_length=256,
                        return_tensors="pt")

        batch = self.tokenizer.build_inputs_for_generation(results, targets=label)
        return batch


def genfiles(path, data):
    s, t = list(zip(*data))
    with open(f'{path}.source', 'a') as f:
        f.write('\n'.join(s))
    with open(f'{path}.target', 'a') as f:
        f.write('\n'.join(t))

if __name__ == '__main__':
    config = yaml.safe_load(open('config.yml'))
    read_json(config['dataset'])
    # lan = args.language
    # samples = config[lan]['samples']
    # OUT_TRAIN = config[lan]['train']
    # OUT_VALID = config[lan]['valid']
    # OUT_TEST = config[lan]['test']

    # convs = json.load(open(samples))
    # samples = []
    # n_train, n_valid, n_test = 0, 0, 0
    # for _, conv in convs.items():
    #     for context in conv["q"]:
    #         for response in conv["a"]:
    #             context = context.replace(' ', '')
    #             response = response.replace(' ', '')
    #             samples.append((f'{context} [sMASK]', response))
    #     if len(samples) > 5000:
    #         random.shuffle(samples)
    #         train = samples[:len(samples) * 48 // 50]
    #         valid = samples[len(samples) * 48 // 50: len(samples) * 49 // 50]
    #         test = samples[len(samples) * 49 // 50: ]
    #         n_train += len(train)
    #         n_valid += len(valid)
    #         n_test += len(test)

    #         genfiles(OUT_TRAIN, train)
    #         genfiles(OUT_VALID, valid)
    #         genfiles(OUT_TEST, test)
    #         samples = []
    # with open(os.path.join(os.path.dirname(f'{OUT_TRAIN}.source'), 'status.tsv'), 'a') as f:
    #     f.write(f'benben faq for glm 10b n_train: {n_train}\tn_valid: {n_valid}\tn_test: {n_test}')
        
        