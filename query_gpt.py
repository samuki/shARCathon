import logging
import copy
# Count tokens before input
import tiktoken

import utils
import config
import llm.prompt_toolkit as prompt_toolkit
import llm.gpt_utils as gpt_utils


def main():
    # Log results
    logger = utils.get_logger()
    # Get token counter
    encoding = tiktoken.encoding_for_model(config.GPT_MODEL)
    folder = str(config.PATH_SELECTION.resolve())
    tasks = utils.load_json_data(folder)
    counter = 0
    for task, value in tasks.items():
        logger.info("TASK %s", task)
        task_name = task.split('.')[0]
        # Copy due to inplace changes
        json_task = copy.deepcopy(value)
        # Merge everything into the prompt
        prompt = gpt_utils.get_prompt(json_task)
        # Replace comma in matrices
        if config.REPLACE_COMMA:
            prompt = prompt.replace(',', '')
        logger.info("PROMPT %s", prompt)
        # Get tokens number due to rate limit 
        encoding_len = len(encoding.encode(prompt))
        logger.info("ENCODING LEN %s", encoding_len)
        # Test if the prompt is too long
        if encoding_len > config.MODEL_MAX_TOKENS - config.MAX_TOKENS:
            logger.info("SKIPPING %s DUE TO ENCODING LEN", task)
            continue
        # Copy due to inplace changes
        json_task = copy.deepcopy(value)
        # Split system and user for for API call
        system = config.PROMPT_TEMPLATE
        user = str(gpt_utils.get_task(json_task))
        # Replace comma in matrices
        if config.REPLACE_COMMA:
            user = user.replace(',', '')
        # Call API
        result = gpt_utils.prompt_gpt(user, system=system)
        logger.info("RESULTS %s", result)
        # Save results
        gpt_utils.save_gpt_results(task_name, prompt, result)
        logger.info("COUNTER %s", counter)
        counter += 1
         

if __name__ == "__main__":
    main()
