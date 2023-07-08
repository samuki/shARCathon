import llm.secret as secret
import config

import openai
import os
import json
import shutil
import re


openai.api_key = secret.API_KEY

COLOR_NUMBER_DICT = {
    "0": "black",
    "1": "white",
    "2": "green",
    "3": "brown",
    "4": "yellow",
    "5": "blue",
    "6": "purple",
    "7": "pink",
    "8": "red",
    "9": "orange"
}

NUMBER_WORD_DICT = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine"
}

NUMBER_BINARY_DICT = {
    "0": "0",
    "1": "1",
    "2": "10",
    "3": "11",
    "4": "100",
    "5": "101",
    "6": "110",
    "7": "111",
    "8": "1000",
    "9": "1001"
}

NUMBER_CHAR_DICT = {
    "0": "a",
    "1": "b",
    "2": "c",
    "3": "d",
    "4": "e",
    "5": "f",
    "6": "g",
    "7": "h",
    "8": "i",
    "9": "j"
}

NUMBER_LEET_DICT = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "E",
    "4": "h",
    "5": "S",
    "6": "b",
    "7": "T",
    "8": "B",
    "9": "g"
}

NUMBER_SP_CHAR_DICT = {
    "0": "!",
    "1": "@",
    "2": "#",
    "3": "$",
    "4": "%",
    "5": "^",
    "6": "&",
    "7": "*",
    "8": "=",
    "9": "+"
}

def prompt_gpt(user, system=False):
    # Use OpenAI API
    # Split system and user for for API call
    if system:
        message = [{"role": "user", "content": user}, {"role": "system", "content": system}]
    else:
        message = [{"role": "user", "content": user}]
    completion = openai.ChatCompletion.create(
        model = config.GPT_MODEL,
        temperature = config.TEMPERATURE,
        max_tokens = config.MAX_TOKENS,
        top_p = config.TOP_P,
        messages = message
    )
    return completion

def preprocess_representation(prompt):
    if config.REPLACE_COMMA:
        prompt = prompt.replace(',', '')
    if config.REPLACE_SPACE:
        while re.search(r'(\d) (\d)', prompt):
            prompt = re.sub(r'(\d) (\d)', r'\1\2', prompt)
    if config.SEMICOLON:
        prompt = prompt.replace('] [', '; ').replace("[[", ";;").replace("]]", ";;")
    if config.BRACKTES:
        prompt = prompt.replace("[", "'").replace("]", "'").replace("[[", "''").replace("]]", "''")
    if config.REPLACE_NUMBER_COLOR:
        for key, value in COLOR_NUMBER_DICT.items():
            prompt = prompt.replace(key, value)
    if config.REPLACE_NUMBER_WORD:
        for key, value in NUMBER_WORD_DICT.items():
            prompt = prompt.replace(key, value)
    if config.REPLACE_NUMBER_BINARY:
        for key, value in NUMBER_BINARY_DICT.items():
            prompt = prompt.replace(key, value)
    if config.REPLACE_NUMBER_CHAR:
        for key, value in NUMBER_CHAR_DICT.items():
            prompt = prompt.replace(key, value)
    if config.REPLACE_NUMBER_LEET:
        for key, value in NUMBER_LEET_DICT.items():
            prompt = prompt.replace(key, value)
    if config.REPLACE_NUMBER_SP_CHAR:
        for key, value in NUMBER_SP_CHAR_DICT.items():
            prompt = prompt.replace(key, value)
    return prompt


def preprocess_prompt(task):
    intro = "Do the following:\nWhat is the step by step description of the input/output relation that holds for all example input/output pairs?\n"
    train_string = "Examples: "
    for example in task['train']:
        train_string += f"input: {str(example['input'])} output: {str(example['output'])} \n"
    divider = "You now have all the information to solve the task. Apply this description to the test input and write you answer as 'output: '\n"
    test_string = f"Test: input: {str(task['test']['input'])} output:"
    return preprocess_representation(intro+ train_string + divider + test_string)


def preprocess_prompt_other(task):
    intro = "Do the following:\nWhat is the step by step pattern that holds for all example input/output pairs?\n"
    train_string = "Examples: "
    for example in task['train']:
        train_string += f"input: {str(example['input'])} output: {str(example['output'])} \n"
    divider = "Apply this pattern to the test input and write the answer as 'output: '\n"
    test_string = f"Test: input: {str(task['test']['input'])} output:"
    return preprocess_representation(intro+ train_string + divider + test_string)
    
def get_task(json_task):
    # ensure only one test output
    json_task['test'] = json_task['test'][0]
    json_task['test']['output'] = ''
    return preprocess_prompt(json_task)


def get_prompt(json_task):
    preamble = config.PROMPT_TEMPLATE
    return preamble + '\n\n' + str(get_task(json_task))


def get_directory():
    dataset = str(config.PATH_SELECTION.resolve()).split('/')[-1]
    replace_comma = 'replace_comma' if config.REPLACE_COMMA else 'no_replace_comma'
    return f"results/{dataset}/{replace_comma}/{config.GPT_MODEL}"


def save_gpt_results(task_name, prompt, result):
    directory = get_directory()
    # check if directory exists and create otherwise
    if not os.path.exists(directory):
        os.makedirs(directory)
    # copy config.py to the new directory
    shutil.copyfile('config.py', os.path.join(directory, 'config.py'))
    # create a json output file
    output_file_name = os.path.join(directory, task_name + "_out.json")
    # data to be written
    data = {"prompt": prompt, "output": result}
    # writing to json file
    with open(output_file_name, 'w') as outfile:
        json.dump(data, outfile)



