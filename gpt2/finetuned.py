#!/usr/bin/env python3

from transformers import pipeline, GPT2LMHeadModel, TrainingArguments, \
    Trainer, TextDataset, DataCollatorForLanguageModeling, GPT2Tokenizer
import os
import random

from . import MODEL, NO_GENERATED_RESULTS, \
    MAX_NO_TOKENS, TOKENIZER, TRAIN_DATA_DIR, DATA_DIR, \
    load_json_data, get_expected_result, select_best_answer
from .prompts import get_prompts
from .output import add_datapoint, dump_data

MODEL_DIR = os.path.abspath("./gpt2/data/gpt2-finetuned-model")
TRAIN_PATH = os.path.abspath("./gpt2/data/gpt2-finetuned-data/train")
TEST_PATH = os.path.abspath("./gpt2/data/gpt2-finetuned-data/test")


def check_model_exists():
    return os.path.exists(MODEL_DIR) and os.path.isdir(MODEL_DIR)


def create_train_data(data, train_path, test_path, kind, list_kind, test_prob=0.15):
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    train_file = open(train_path, 'w+')
    test_file = open(test_path, 'w+')
    for value in data.values():
        prompts = get_prompts(value, kind=kind, list_kind=list_kind)
        exp_result = get_expected_result(value, list_kind=list_kind)
        for prompt in prompts:
            is_test = random.random() <= test_prob
            f = train_file if not is_test else test_file
            result = prompt + exp_result
            f.write(result)


def train(kind, list_kind, data):
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL)
    model = GPT2LMHeadModel.from_pretrained(MODEL)
    create_train_data(data, TRAIN_PATH, TEST_PATH, kind, list_kind)

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,  # The output directory
        overwrite_output_dir=True,  # overwrite the content of the output dir
        num_train_epochs=3,  # number of training epochs
        per_device_train_batch_size=8,  # batch size for training
        per_device_eval_batch_size=16,  # batch size for evaluation
        eval_steps=400,  # Number of update steps between two evaluations.
        save_steps=800,  # after # steps model is saved
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
    )

    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=TRAIN_PATH,
        block_size=128)

    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=TEST_PATH,
        block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Start training
    trainer.train()

    # Save model for reuse later on
    trainer.save_model()


def basic_generator(generator, prompt, list_kind='small', max_len=MAX_NO_TOKENS):
    results = generator(
        prompt, num_return_sequences=NO_GENERATED_RESULTS, max_length=max_len)
    answers = [
        result['generated_text'].replace(prompt, '', 1)
        for result in results
    ]
    result = select_best_answer(answers, list_kind)
    return result


def main(json_path, kind='basic', list_kind='small'):
    global MODEL_DIR, TRAIN_PATH, TEST_PATH
    MODEL_DIR = os.path.abspath(f"./gpt2/data/gpt2-finetuned-{kind}-{list_kind}-model")
    TRAIN_PATH = os.path.abspath(f"./gpt2/data/gpt2-finetuned-{kind}-{list_kind}-data/train")
    TEST_PATH = os.path.abspath(f"./gpt2/data/gpt2-finetuned-{kind}-{list_kind}-data/test")

    data = load_json_data(DATA_DIR)
    train_data = load_json_data(TRAIN_DATA_DIR)
    if not check_model_exists():
        train(kind, list_kind, train_data)

    generator = pipeline('text-generation', model=MODEL, device="cuda:0")

    for task, value in data.items():
        print(f"\t|> Task: {task}")
        prompts = get_prompts(value, kind=kind, list_kind=list_kind)
        exp_result = get_expected_result(value, list_kind)
        for prompt in prompts:
            print(f"\t|> Prompt: \n{prompt}")
            result = basic_generator(generator, prompt, list_kind=list_kind)
            print(f"\t|> Result: \n{result}")
            print(f"\t|> Expected Result: \n{exp_result}")
            add_datapoint(prompt, result, exp_result)
    dump_data(json_path)
