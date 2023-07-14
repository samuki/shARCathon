# shARCathon
<img align="left" src="images/sharcathon.png" width="250" alt="">

## Project Structure

```
.
├──── cnn : cnn baseline model
├──── data : extract ARC-800 zip here
├──── examples : contains example images of tasks and cnn output
├──── gpt2 : pipeline to finetune gpt2 + measurements
├──── gptj : pipeline to finetune gptj
├──── gptneo : pipeline to finetune gptneo + measurements
├──── images : images used in readme + report
├──── llm : 
├──── results : all output files of all models and all prompting techniques
├──── config.py : contains information for data splits
├──── dataset_exploration.ipynb : notebook used for data exploration
├──── evaluate_local_models.py : evalaute local models using our evaluation metrics
├──── filter_data.py : run this to create the data splits after extracting the zip in data
├──── query_gpt.py : make gpt api calls and save results
├──── run_local_models.sh : run model and prompting technique
├──── task_evaluation.ipynb : visualize different tasks + task statistics
├──── task.json : contains manual mapping of task labels to tasks
├──── train_cnn.py : train cnn baseline model
├──── utils.py : util functions
```

## Requirements

To run all of our code following requirements have to be met:
 - Have a valid API Key from OpenAI
 - Have a GPU with at least 30 Gigabyte of VRAM
 - python3.10+

## Setup

To get started make sure you installed all the requirements using pip.
```bash
pip install -r requirements.txt
```

To prepare our test data we recommend you to run the following commands in your shell:
```bash
unzip ./data/ARC-800-tasks.zip -d ./data
python filter_data.py
mv ./data/evaluation ./data/evaluation_orig
mv ./data/evaluation_small ./data/evaluation
```

Additionally our GPT3 and 4 code relies on the OpenAI API.
Therefore make sure to create `llm/secret.py` with following content:
```python
API_KEY = "<Your-API-key>"
```

Also ensure that you have adapted the `config.py` to your wanted parameters.

## Running GPT-2, GPT Neo, GPT-J

We have prepared a small helper script that starts all possible inference types for GPT-2, Neo and J.
For that run:
```bash
bash run_local_models.sh
```

After that you can run following command to evaluate the output of the inference:
```bash
python evaluate_local_models.py
```

## Running GPT-3, GPT-3.5, GPT-4

### Representation Analysis
```bash
python query_gpt.py
```

### Self Consistency Evaluation
```bash
python query_gpt_task_2.py
```


## Appendix
For ease of understanding the outputs, we have accumulated some additional example output and the hyperparameters used for finetuning.

### Prior Prompt
```
You are given a series of inputs and output pairs. 
    These are all in the form of a 2D array, representing a 2D grid, with values from 0-9. 
    The values are not representative of any ordinal ranking. 
    Input/output pairs may not reflect all possibilities, you are to infer the simplest possible relation making use of symmetry and invariance as much as possible.

    The input can be something like:
    > entire grid being the sandbox to manipulate
    > using a part of the grid (individual squares or portions of the grid) to depict instructions of how to do the task. Position and symmetry is very important.
    > using regions of similar value to depict area for answer of the task

    The output can be something like:
    > same output size as input after performing action
    > output one of the fixed predetermined patterns used to classify the input image
    > using output to show the ordering of objects, such as by size, height, width, position, value

    Each of the input-output relation can be done with one or more actions chained together, which could be something like (not exhaustive):
    - object view (defined as continuous squares connected horizontally, vertically and/or diagonally, separated by 0 values)
    > objects can be the of the same value, or different values combined together
    > objects may be hidden beneath other objects
    > rotating or shifting objects
    > changing value of object
    > objects can be manipulated and mapped to a different number of output squares
    > different objects may be manipulated differently based on context

    - pixel view
    > rotation / reflection symmetry
    > continuation of a pattern
    > changing values

    - segment view
    > combine two segments of the input into one single one based on a simple rule
    > rule can be certain values are prioritized over others, or combination of values into new ones

    Do the following:
    - What is the broad description of the input/output relation that holds for all input/output pairs?
    - What is the step by step description of the input/output relation that holds for all input/output pairs? 
    - Apply this description to the test input and find out the answer 'to_be_filled'.

{'train': [{'input': [[3, 1, 2], [3, 1, 2], [3, 1, 2]], 'output': [[4, 5, 6], [4, 5, 6], [4, 5, 6]]}, {'input': [[2, 3, 8], [2, 3, 8], [2, 3, 8]], 'output': [[6, 4, 9], [6, 4, 9], [6, 4, 9]]}, {'input': [[5, 8, 6], [5, 8, 6], [5, 8, 6]], 'output': [[1, 9, 2], [1, 9, 2], [1, 9, 2]]}, {'input': [[9, 4, 2], [9, 4, 2], [9, 4, 2]], 'output': [[8, 3, 6], [8, 3, 6], [8, 3, 6]]}], 'test': {'input': [[8, 1, 3], [8, 1, 3], [8, 1, 3]], 'output': 'to_be_filled'}}
```

### Basic Prompt
```
Continue the pattern
Examples: input: [[6, 0, 0, 4, 0, 0, 8], [0, 6, 0, 4, 0, 0, 8], [0, 6, 0, 4, 8, 8, 0]] output: [[2, 0, 2], [0, 2, 2], [2, 2, 0]] 
input: [[0, 0, 6, 4, 8, 8, 0], [0, 6, 0, 4, 0, 8, 8], [0, 6, 6, 4, 8, 0, 0]] output: [[2, 2, 2], [0, 2, 2], [2, 2, 2]] 
input: [[0, 0, 6, 4, 8, 0, 8], [6, 0, 6, 4, 0, 0, 0], [0, 6, 6, 4, 8, 0, 8]] output: [[2, 0, 2], [2, 0, 2], [2, 2, 2]] 
input: [[6, 0, 6, 4, 0, 0, 0], [6, 6, 0, 4, 8, 0, 8], [6, 6, 6, 4, 0, 8, 0]] output: [[2, 0, 2], [2, 2, 2], [2, 2, 2]] 
input: [[0, 0, 6, 4, 8, 0, 8], [0, 6, 0, 4, 0, 8, 0], [0, 0, 0, 4, 8, 0, 0]] output: [[2, 0, 2], [0, 2, 0], [2, 0, 0]]
Test: input: [[0, 6, 6, 4, 0, 0, 8], [0, 6, 0, 4, 8, 8, 8], [6, 0, 6, 4, 0, 0, 0]] output:
```


### Step-by-step Prompt
```
Do the following:
What is the step-by-step description of the input/output relation that holds for all example input/output pairs?
Examples: input: [[6, 0, 0, 4, 0, 0, 8], [0, 6, 0, 4, 0, 0, 8], [0, 6, 0, 4, 8, 8, 0]] output: [[2, 0, 2], [0, 2, 2], [2, 2, 0]] 
input: [[0, 0, 6, 4, 8, 8, 0], [0, 6, 0, 4, 0, 8, 8], [0, 6, 6, 4, 8, 0, 0]] output: [[2, 2, 2], [0, 2, 2], [2, 2, 2]] 
input: [[0, 0, 6, 4, 8, 0, 8], [6, 0, 6, 4, 0, 0, 0], [0, 6, 6, 4, 8, 0, 8]] output: [[2, 0, 2], [2, 0, 2], [2, 2, 2]] 
input: [[6, 0, 6, 4, 0, 0, 0], [6, 6, 0, 4, 8, 0, 8], [6, 6, 6, 4, 0, 8, 0]] output: [[2, 0, 2], [2, 2, 2], [2, 2, 2]] 
input: [[0, 0, 6, 4, 8, 0, 8], [0, 6, 0, 4, 0, 8, 0], [0, 0, 0, 4, 8, 0, 0]] output: [[2, 0, 2], [0, 2, 0], [2, 0, 0]] 
You now have all the information to solve the task. Apply this description to the test input and write you answer as 'output: '
Test: input: [[0, 6, 6, 4, 0, 0, 8], [0, 6, 0, 4, 8, 8, 8], [6, 0, 6, 4, 0, 0, 0]] output:
```


### Hyperparameters

#### GPT-2, GPT Neo and GPT J Finetuning Parameters
```
num_train_epochs            = 3
per_device_train_batch_size = 8
per_device_eval_batch_size  = 16
eval_steps                  = 400
save_steps                  = 800
warmup_steps                = 500
```

#### GPT-2, GPT Neo and GPT J Interference Parameters
```
max_length  = MAX_NO_TOKENS # This variable should be set to the context size of the model
temperature = 0.01          # Only set for GPT-J available
```

#### GPT3.5 and 4
```
top_p       = 1
temperature = 0
max_tokens  = 4000
```






