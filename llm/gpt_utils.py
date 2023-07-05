import llm.secret as secret
import config

import openai
import os
import json
import shutil


openai.api_key = secret.API_KEY


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


def preprocess(task):
    intro = "Do the following:\nWhat is the step by step description of the input/output relation that holds for all example input/output pairs?\n"
    train_string = "Examples: "
    for example in task['train']:
        train_string += f"input: {str(example['input'])} output: {str(example['output'])} \n"
    divider = "You now have all the information to solve the task. Apply this description to the test input and write you answer as 'output: '\n"
    test_string = f"Test: input: {str(task['test']['input'])} output:"
    
    return intro+train_string + divider + test_string
    
    
def get_task(json_task):
    # ensure only one test output
    json_task['test'] = json_task['test'][0]
    json_task['test']['output'] = ''
    return preprocess(json_task)


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


