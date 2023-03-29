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

def train():
    dataset = load_dataset("BelleGroup/generated_train_0.5M_CN")
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
    train_dataset = dataset['train']

    def encoded_dataset(samples):
        sentences = []
        labels = []
        for sample in zip(samples['input'], samples['target']):
            sentence = sample[0].replace('\n', '').replace('\\', '')
            label = sample[1]

            sentence = f'### 指令: {sentence} <\s> ### 回答: '
            sentences.append(sentence)
            labels.append(f'{label}')

        results = tokenizer(sentences,
                        padding="max_length",
                        truncation=True,
                        max_length=128)
        labels = tokenizer(labels,
                        padding="max_length",
                        truncation=True,
                        max_length=256)
        results["labels"] = labels["input_ids"]
        return results

    train_dataset = train_dataset.map(encoded_dataset, batched=True, remove_columns=['input', 'target'])
    
    chkpt_path = 'chkpt'

    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                                model=model,
                                                return_tensors="pt")

    training_args = Seq2SeqTrainingArguments(
        learning_rate=2e-4,
        save_steps=3000,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        predict_with_generate=True,
        fp16=True,
        save_total_limit=20,
        output_dir=chkpt_path, #The output directory
        overwrite_output_dir=False, #overwrite the content of the output directory
        num_train_epochs=3, # number of training epochs
        deepspeed="config_deepspeed.json"
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    logger.info('开始训练')
    trainer.train()

    logger.info('保存模型')
    trainer.save_model()

if __name__ == '__main__':
    train()

