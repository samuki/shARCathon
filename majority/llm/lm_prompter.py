import sys
# Get access to the utils from one folder above
sys.path.append('../')
import utils

import gpt_utils
import copy


def main():
    folder = '../data/evaluation'
    myjson = utils.load_json_data(folder)
    task_name = 'ca8de6ea'
    json_task = copy.deepcopy(myjson[task_name +'.json'])

    #prompt = gpt_utils.get_prompt(json_task).replace(',', '')
    #print(prompt)
    
    json_task = copy.deepcopy(myjson[task_name +'.json'])

    # Add GPT output here (if any)
    #json_task['gpt_output'] = [[9, 5, 4], [9, 5, 4], [9, 5, 4]]

    json_task['gpt_output'] = None
    print(json_task)
    utils.plot_single_output(task_name,"GPT-4", json_task)
    #utils.plot_2d_grid(task_name,"GPT-4", json_task)

if __name__ == "__main__":
    main()