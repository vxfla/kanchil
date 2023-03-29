from transformers import MT5Tokenizer, MT5ForConditionalGeneration, DataCollatorForSeq2Seq
from transformers import WEIGHTS_NAME, CONFIG_NAME
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import torch.utils.data as Data
import torch.optim as optim
import torch
import yaml
import os
import time


def test():
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
    model = MT5ForConditionalGeneration.from_pretrained("/users12/rzliu/workspace/chatMT5base_zh/chkpt")
    model = model
    model.eval()

    while True:
        q = input('user > ')

        inputs = tokenizer(f"### 指令: {q} <\s> ### 回答: ", return_tensors="pt", padding=False)

        inputs = inputs
        outputs = model.generate(inputs.input_ids, max_length=512, 
                                    eos_token_id=tokenizer.eos_token_id, num_beams=3,
                                     no_repeat_ngram_size=3, top_p=0.7, temperature=2.0)
        print(f'chatT5 base > {tokenizer.decode(outputs[0, :].tolist(), skip_special_tokens=True)}')

    
if __name__ == '__main__':
    test()
