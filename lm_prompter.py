import utils
import gpt_utils
import copy


def main():
    folder = 'data/training'
    myjson = utils.load_json_data(folder)
    task_name = '0d3d703e'
    json_task = copy.deepcopy(myjson[task_name +'.json'])

    prompt = gpt_utils.get_prompt(json_task)
    print(prompt)
    
    json_task = copy.deepcopy(myjson[task_name +'.json'])

    # Add GPT output here (if any)
    json_task['gpt_output'] = [[9, 5, 4], [9, 5, 4], [9, 5, 4]]

    # json_task['gpt_output'] = None

    utils.plot_single_output(task_name,"GPT-4", json_task)
    utils.plot_2d_grid(task_name,"GPT-4", json_task)

if __name__ == "__main__":
    main()