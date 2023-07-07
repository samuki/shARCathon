#!/usr/bin/env python3

from transformers import pipeline, GPT2LMHeadModel, TrainingArguments, \
    Trainer, TextDataset, DataCollatorForLanguageModeling, AutoTokenizer
import numpy as np
import os
import random

from . import \
    DEBUG, MODEL, NO_GENERATED_RESULTS, \
    MAX_NO_TOKENS, TOKENIZER, DATA_DIR, \
    load_json_data, get_expected_result, \
    get_logger
from .prompts import get_prompts


def check_model_exists(kind):
    model_dir = f"./gpt2/data/gpt2-finetuned-{kind}-model"
    return os.path.exists(model_dir) and os.path.isdir(model_dir)


def create_train_data(data, train_path, test_path, kind, test_prob=0.15):
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    train_file = open(train_path, 'w+')
    test_file = open(test_path, 'w+')
    for value in data.values():
        prompts = get_prompts(value, kind=kind)
        exp_result = get_expected_result(value)
        for prompt in prompts:
            is_test = random.random() <= test_prob
            f = train_file if not is_test else test_file
            result = prompt + exp_result
            f.write(result)


def train(kind, data):
    model_dir = f"./gpt2/data/gpt2-finetuned-{kind}-model"
    train_path = f"./gpt2/data/gpt2-finetuned-{kind}-data/train"
    test_path = f"./gpt2/data/gpt2-finetuned-{kind}-data/test"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = GPT2LMHeadModel.from_pretrained(MODEL)
    create_train_data(data, train_path, test_path, kind)

    training_args = TrainingArguments(
        output_dir=model_dir,  # The output directory
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
        file_path=train_path,
        block_size=128)

    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
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
        # prediction_loss_only=True,
    )

    # Start training
    trainer.train()

    # Save model for reuse later on
    trainer.save_model()


EXPECTED_CHARS = [str(i) for i in range(0, 10)] + [';', ' ']


def select_best_answer(answers):
    if len(answers) <= 0:
        return None
    if len(answers) == 1:
        return answers[0]

    # Simple approach: Pick the answer with the least 'non-expected chars'
    scores = len(answers) * [0]
    for i, answer in enumerate(answers):
        score = 0
        for c in answer:
            if c not in EXPECTED_CHARS:
                score += 1
        scores[i] = score
    return answers[np.argmin(scores)]


def basic_generator(prompt, kind='basic', max_len=MAX_NO_TOKENS):
    model_dir = f"./gpt2/data/gpt2-finetuned-{kind}-model"
    generator = pipeline('text-generation', model=model_dir)
    results = generator(
        prompt, num_return_sequences=NO_GENERATED_RESULTS, max_length=max_len)
    answers = [result['generated_text'].removeprefix(
        prompt) for result in results]
    result = select_best_answer(answers)
    return result


def main(kind='basic'):
    logger = get_logger()
    if DEBUG:
        logger.info = print

    data = load_json_data(DATA_DIR)
    if not check_model_exists(kind):
        train(kind, data)

    for task, value in data.items():
        logger.info(f"\t|> Task: {task}")
        prompts = get_prompts(value, kind=kind)
        exp_result = get_expected_result(value)
        for prompt in prompts:
            logger.info(f"\t|> Prompt: \n{prompt}")
            no_tokens = len(TOKENIZER(prompt)['input_ids']) \
                + len(TOKENIZER(exp_result)['input_ids'])
            result = basic_generator(prompt, kind=kind, max_len=no_tokens)
            logger.info(f"\t|> Result: \n{result}")
