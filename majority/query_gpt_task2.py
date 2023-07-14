import copy
# Count tokens before input
import tiktoken
from pathlib import Path

import utils
import config
import llm.gpt_utils as gpt_utils    
import json
import os
import random 
    

def task2_majority():
    # Log results
    logger = utils.get_logger()
    # Get token counter
    encoding = tiktoken.encoding_for_model(config.GPT_MODEL)

    folder = str(config.PATH_SELECTION.resolve())
    tasks = utils.load_json_data(folder)

    for task, value in tasks.items():

        task_name = task.split('.')[0]

        # Copy due to inplace changes
        json_task = copy.deepcopy(value)
        path_to_save = os.path.join("task2_false_majority", task_name + "_out.json")

        if os.path.exists(path_to_save):
            print(f"{task_name} continuing")
            continue

        solution = json_task["test"][0]['output']

        # modify solution
        dim1 = len(solution)
        dim2 = len(solution[0])

        r1 = random.randint(0, dim1-1)
        r2 = random.randint(0, dim2-1)

        if solution[r1][r2] == 9: 
            solution[r1][r2] -= 1
        else: 
            solution[r1][r2] += 1
        
        # Merge everything into the prompt
        # Split system and user for for API call
        part_1 = str(gpt_utils.get_task2(json_task))

        complete_prompt = part_1 + f' \nTest Output {solution}'

        result_list = []

        # Copy due to inplace changes
        json_task = copy.deepcopy(value)

        for i in range(5):

            # Call API
            result = gpt_utils.prompt_gpt(complete_prompt)
            
            string = result['choices'][0]['message']['content'].lower()
            yes_answer = False
            no_answer = False

            if "yes" in string: 
                yes_answer = True

            if "no" in string: 
                no_answer = True 

            # heuristic to find confidence level based on % 
            # TODO: find better solution
            try:
                confidence_level = ""
                confidence_split = string.split("%")[0]
                index = 1
                while confidence_split[-index] != " ":
                    confidence_level = confidence_split[-index] + confidence_level
                    index += 1
                
            except:
                confidence_level = None

            print("yes_answer", yes_answer, "no_answer", no_answer, "confidence", confidence_level)
            
            result_list.append({ "answer": result, "yes_answer": yes_answer, "no_answer": no_answer, "confidence": confidence_level })

        print("\n")
        with open(path_to_save, "w") as f:
            f.write(json.dumps(result_list))
        
        

def task_2():
    # Log results
    logger = utils.get_logger()
    # Get token counter
    encoding = tiktoken.encoding_for_model(config.GPT_MODEL)

    folder = str(config.PATH_SELECTION.resolve())
    tasks = utils.load_json_data(folder)
    counter = 0
    for task, value in tasks.items():

        task_name = task.split('.')[0]

        # Copy due to inplace changes
        json_task = copy.deepcopy(value)
        path_to_save = os.path.join("task_2", task_name + "_out.json")

        if os.path.exists(path_to_save):
            print(f"{task_name} continuing")
            continue

        solution = json_task["test"][0]['output']

        """
        # modify solution
        dim1 = len(solution)
        dim2 = len(solution[0])

        r1 = random.randint(0, dim1-1)
        r2 = random.randint(0, dim2-1)

        solution[r1][r2] += 1
        """

        # Merge everything into the prompt
        # Split system and user for for API call
        part_1 = str(gpt_utils.get_task2(json_task))

        complete_prompt = part_1 + f' \nTest Output {solution}'

        encoding_len = len(encoding.encode(complete_prompt))

        print(complete_prompt)

        # Get tokens number due to rate limit 
        # Test if the prompt is too long
        if encoding_len > config.MODEL_MAX_TOKENS - config.MAX_TOKENS:
            logger.info("SKIPPING %s DUE TO ENCODING LEN", task)
            continue
        # Copy due to inplace changes
        json_task = copy.deepcopy(value)

        # Call API
        result = gpt_utils.prompt_gpt(complete_prompt)
        
        print("result", result['choices'][0]['message']['content'])
        print("\n\n -----")

        string = result['choices'][0]['message']['content'].lower()
        yes_answer = False
        no_answer = False

        if "yes" in string: 
            yes_answer = True

        if "no" in string: 
            no_answer = True 

        # heuristic to find confidence level based on % 
        # TODO: find better solution
        try:
            confidence_level = ""
            confidence_split = string.split("%")[0]
            index = 1
            while confidence_split[-index] != " ":
                confidence_level = confidence_split[-index] + confidence_level
                index += 1
            
        except:
            confidence_level = None

        print("yes_answer", yes_answer, "no_answer", no_answer, "confidence", confidence_level)
        
        result_object = { "answer": result, "yes_answer": yes_answer, "no_answer": no_answer, "confidence": confidence_level }

        with open(path_to_save, "w") as f:
            f.write(json.dumps(result_object))
        


if __name__ == "__main__":
    task2_majority()



