import copy
# Count tokens before input
import tiktoken
from pathlib import Path

import utils
import config
import llm.gpt_utils as gpt_utils
    

def main():
    # Log results
    logger = utils.get_logger()
    # Get token counter
    encoding = tiktoken.encoding_for_model(config.GPT_MODEL)
    folder = str(config.PATH_SELECTION.resolve())
    tasks = utils.load_json_data(folder)
    training_tasks = utils.load_json_data(str(config.TRAIN_SMALL_PATH))
    counter = 0
    for task, value in tasks.items():
        logger.info("TASK %s", task)
        task_name = task.split('.')[0]
        # Check if output has alrerady been generated
        if Path(gpt_utils.get_directory()+"/"+task_name+"_out.json").is_file():
            logger.info("TASK %s ALREADY EXISTS", task)
            continue
        # Copy due to inplace changes
        json_task = copy.deepcopy(value)
        # Merge everything into the prompt
        # Split system and user for for API call
        system = config.PROMPT_TEMPLATE
        user = str(gpt_utils.get_task(json_task))
        if config.ADD_TRAINING:
            prompt = ""
            for train_task_name in config.SELECTED_TRAINING_TASKS:
                train_task = copy.deepcopy(training_tasks[train_task_name])
                training_prompt = str(gpt_utils.get_task(train_task, training=True))
                prompt += training_prompt 
            prompt +=  system+ user
        else:
            prompt = system + user
        # Replace comma in matrices
        
        logger.info("PROMPT %s", prompt)
        # Get tokens number due to rate limit 
        encoded = encoding.encode(prompt)
        encoded_len = len(encoded)
        
        
        logger.info("ENCODING LEN %s", encoded_len)
        # Test if the prompt is too long
        if encoded_len > config.MODEL_MAX_TOKENS - config.MAX_TOKENS:
            logger.info("SKIPPING %s DUE TO ENCODING LEN", task)
            continue
        # Copy due to inplace changes
        json_task = copy.deepcopy(value) 
        print(prompt)
        # Call API
        result = gpt_utils.prompt_gpt(user+system)
        print(f'First {gpt_utils.naive_postprocessing(gpt_utils.extract_result_text(result))}')
        if config.SELF_CONSISTENCY:
            prediction = gpt_utils.naive_postprocessing(gpt_utils.extract_result_text(result))
            user = str(gpt_utils.get_task(json_task, self_correction=True))
            prompt = system+user+f"{prediction}. Wait, no. 'output: '"
            logger.info("SELF CONSISTENCY PROMPT %s", prompt)
            result = gpt_utils.prompt_gpt(prompt)
            print(f'Second {gpt_utils.naive_postprocessing(gpt_utils.extract_result_text(result))}')
        if config.DOUBLE_SELF_CONSISTENCY:
            prediction1 = gpt_utils.naive_postprocessing(gpt_utils.extract_result_text(result))
            result2 = gpt_utils.prompt_gpt(user+system)
            print(f'Second {gpt_utils.naive_postprocessing(gpt_utils.extract_result_text(result2))}')
            prediction2 = gpt_utils.naive_postprocessing(gpt_utils.extract_result_text(result2))
            user = str(gpt_utils.get_task(json_task, self_correction=True))
            prompt = system+user+f"{prediction1} output 2:{prediction2}\nFinal output:"
            logger.info("SELF CONSISTENCY PROMPT %s", prompt)
            result = gpt_utils.prompt_gpt(prompt)
            print(f'Third {gpt_utils.naive_postprocessing(gpt_utils.extract_result_text(result2))}')

        logger.info("RESULTS %s", result)
        # Save results
        gpt_utils.save_gpt_results(task_name, prompt, result)
        logger.info("COUNTER %s", counter)
        counter += 1
        
        
if __name__ == "__main__":
    main()
