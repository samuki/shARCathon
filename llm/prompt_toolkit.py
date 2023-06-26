# Collection of prompts

PREMEABLE = '''You are given a series of inputs and output pairs. 
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
    - Apply this description to the test input and find out the answer 'to_be_filled'.'''
    
BETTER_JSON_PREMEABLE = '''You are given a series of inputs and output pairs. 
    These are in the form of a 2D array, representing a 2D grid, with values from 0-9. 
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
    You will recieve a json file with the following format:
    {'train': [{'input': [], 'output': [], {'input': [], 'output': []}, ...], 'output': [], 
    'test': {'input':[], 'step_by_step': 'step_by_step_to_be_filled, 'output': 'output_to_be_filled'}}
    Do the following:
    - Start by filling in 'step_by_step' with the reasoning on the pattern input/output relation that holds for all input/output pairs.
    - Fill in 'output_to_be_filled' by applying reasoning in step_by_step_to_be_filled to the test input.'''
