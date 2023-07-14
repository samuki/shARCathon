import copy
import json
import utils

import llm.prompt_toolkit as prompt_toolkit
import llm.gpt_utils as gpt_utils


def main():
    folder = 'results/gpt-4/'
    taskname = "0a2355a6"
    task = taskname + "_out.json"
    with open(folder + task, "r") as f:
        read = json.load(f)
    print(read)
    """
    tasks = utils.load_json_data(folder)
    for task, value in tasks.items():
        task_name = task.split('.')[0]
        json_task = copy.deepcopy(value)
        system = prompt_toolkit.BETTER_JSON_PREMEABLE
        user = str(gpt_utils.get_task(json_task))
        with open(f"results/gpt-4/{task_name}_out.json", "w") as f:
            json.dump({"prompt": system+user, "output": ""}, f)
    """
if __name__ == "__main__":
    main()