import logging
from datetime import datetime

import utils
import copy
import config
import llm.prompt_toolkit as prompt_toolkit
import llm.gpt_utils as gpt_utils

def main():
    # set up logging to file - see previous section for more details
    now = datetime.now() # current date and time
    log_name = now.strftime("%Y%m%d%H%M%S")
    log_name = f"{log_name}.log"
    logging.basicConfig(level=logging.INFO, filename=log_name, filemode='w', 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    
    folder = 'data/evaluation'
    tasks = utils.load_json_data(folder)
    counter = 0
    for task in tasks:
        task_name = task.split('.')[0]
        json_task = copy.deepcopy(tasks[task])
        # merge everything into the prompt
        prompt = gpt_utils.get_prompt(json_task)
        json_task = copy.deepcopy(tasks[task])
        # split system and user
        system = prompt_toolkit.BETTER_JSON_PREMEABLE
        user = str(gpt_utils.get_task(json_task))
        logger.info("TASK %s", task)
        logger.info("COUNTER %s", counter)
        logger.info("PROMPT %s", prompt)
        result = gpt_utils.prompt_gpt(user, system=system)
        logger.info("RESULTS %s", result)
        gpt_utils.save_gpt_results(task_name, prompt, result)
        counter += 1

if __name__ == "__main__":
    main()
