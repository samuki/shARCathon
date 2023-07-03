import llm.secret as secret
import llm.prompt_toolkit as prompt_toolkit
import config

import openai
import os
import json
import shutil


openai.api_key = secret.API_KEY


def prompt_gpt(user, system=False):
    # Use OpenAI API
    if system:
        message = [{"role": "user", "content": user}, {"role": "system", "content": system}]
    else:
        message = [{"role": "user", "content": user}]
    completion = openai.ChatCompletion.create(
        model = config.GPT_MODEL,
        temperature = config.TEMPERATURE,
        max_tokens = config.MAX_TOKENS,
        messages = message
    )
    return completion


def get_task(json_task):
    # ensure only one test output
    json_task['test'] = json_task['test'][0]
    json_task['test']['step_by_step'] = 'step_by_step_to_be_filled'
    json_task['test']['output'] = 'output_to_be_filled'
    return json_task


def get_prompt(json_task):
    preamble = prompt_toolkit.BETTER_JSON_PREMEABLE
    return preamble + '\n\n' + str(get_task(json_task))


def save_gpt_results(task_name, prompt, result):
    directory = "results/{}".format(config.GPT_MODEL)
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



