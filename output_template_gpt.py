import copy
import json
import utils

import llm.prompt_toolkit as prompt_toolkit
import llm.gpt_utils as gpt_utils


def main():
    folder = 'data/evaluation'
    tasks = utils.load_json_data(folder)
    for task, value in tasks.items():
        # Merge everything into the prompt
        json_task = copy.deepcopy(value)
        # Split system and user for for API call
        system = prompt_toolkit.BETTER_JSON_PREMEABLE
        user = str(gpt_utils.get_task(json_task))
        with open(f"results/gpt-4/{task}", "w") as f:
            json.dump({"prompt": system+user, "output": ""}, f)
         

if __name__ == "__main__":
    main()
