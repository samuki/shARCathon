import copy
import utils

import gpt_utils


def main():
    folder = '../data/evaluation'
    tasks = utils.load_json_data(folder)
    for task, value in tasks.items():
        task_name = task.split('.')[0]
        json_task = copy.deepcopy(value)
        prompt = gpt_utils.get_prompt(json_task)
        with open(f"results/gpt-4/{task_name}_out.txt", "w") as f:
            f.write("Prompt:\n")
            f.write(prompt.replace(',', ''))
            f.write("\n\nOutput:\n")
        
         
if __name__ == "__main__":
    main()
