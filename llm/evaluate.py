import json
import re
from pathlib import Path
import sys

sys.path.append('../')
import gpt_utils
import config


def format_percentage(percentage):
    """Formats percentage with two decimal points"""
    return f"{percentage * 100:.1f}"


def postprocess_representation(representation):
    """Post-processes the model representation"""
    if config.REPLACE_NUMBER_COLOR:
        for key, value in {v: k for k, v in gpt_utils.COLOR_NUMBER_DICT.items()}.items():
            representation = representation.replace(key, value)
    if config.REPLACE_NUMBER_WORD:
        for key, value in {v: k for k, v in gpt_utils.NUMBER_WORD_DICT.items()}.items():
            representation = representation.replace(key, value)
    if config.REPLACE_SPACE:
        representation = re.sub(r'(?<=\d)(?=\d)', ' ', representation)
    if config.REPLACE_COMMA:
        representation = representation.replace(' ', ', ')
    if config.REPLACE_NUMBER_CHAR:
        representation = re.sub(r'([a-z]+)', r'"\1"', representation)
    if config.SEMICOLON:
        representation = representation.replace(';', ':')
    return json.loads(representation)


def naive_postprocessing(prediction):
    """Performs simple post-processing on model prediction"""
    return prediction.split(':')[-1].replace("\n", "").strip()


def evaluate_model_performance(ground_truth_folder, model_predictions_folder):
    """Evaluate the performance of language model"""
    task_files = list(ground_truth_folder.glob('*.json'))
    correct_dimensions, correct_predictions, correct_cells, total_predictions, total_cells, all_correct = 0, 0, 0, 0, 0, []

    for task_file in task_files:
        prediction_file = model_predictions_folder / f"{task_file.stem}_out.json"
        if prediction_file.exists():
            with open(prediction_file, 'r') as pred_file, open(task_file, 'r') as gt_file:
                prediction = json.load(pred_file)
                ground_truth = json.load(gt_file)

            ground_truth_out = str(ground_truth['test'][0]['output'])
            ground_truth_out = gpt_utils.preprocess_representation(ground_truth_out)
            result = prediction['output']["choices"][0]["message"]["content"]
            postprocessed_result = naive_postprocessing(result)
            #print(postprocessed_result)
            try:
                json_postprocessed_result = postprocess_representation(postprocessed_result)
            except json.decoder.JSONDecodeError:
                json_postprocessed_result = [[]]
            json_ground_truth_out = postprocess_representation(ground_truth_out)
            print(f"Postprocessed ground truth {json_ground_truth_out}")
            print(f"Postprocessed prediction {json_postprocessed_result}")
            if len(json_postprocessed_result) == len(json_ground_truth_out) and all(
                    len(pred_row) == len(gt_row) for pred_row, gt_row in zip(json_postprocessed_result, json_ground_truth_out)):
                correct_dimensions += 1
                for pred_row, gt_row in zip(json_postprocessed_result, json_ground_truth_out):
                    for pred_cell, gt_cell in zip(pred_row, gt_row):
                        total_cells += 1
                        if pred_cell == gt_cell:
                            correct_cells += 1
            if ground_truth_out == postprocessed_result:
                correct_predictions += 1
                all_correct.append(task_file.name)
            total_predictions += 1
        else:
            print(f'Task {task_file.name}: No prediction file found.')
    return correct_dimensions, correct_predictions, correct_cells, total_predictions, total_cells, all_correct


if __name__ == "__main__":
    model_name = "gpt-3.5-turbo-16k"
    #model_name = "gpt-4"
    #replace_comma = "no_replace_comma"
    replace_comma = "replace_comma"
    ground_truth_folder = Path("../data/evaluation_small")
    predictions_base_folder = Path("../results/evaluation_small_colors_gpt3.5_gpt4") # colors
    predictions_base_folder = Path("../results/evaluation_small_gpt4_replace_comma") 
    predictions_base_folder = Path("../results/evaluation_small_double_semicolon")
    predictions_base_folder = Path("../results/evaluation_small_gpt_4_spaces")
    predictions_base_folder = Path("../results/evaluation_small_no_comma_gpt-3.5_comma_gpt4")
    predictions_base_folder = Path("../results/copy_better_prompt_evaluation_small") 
    predictions_base_folder = Path("../results/evaluation_small") # no space
    #predictions_base_folder = "../results/better_prompt_evaluation_small"
    #predictions_base_folder = Path("../results/evaluation_small")
    model_predictions_folder = predictions_base_folder / replace_comma / model_name

    dim_correct, acc_correct, c_acc_correct, total, c_acc_total, all_correct = evaluate_model_performance(ground_truth_folder, model_predictions_folder)

    cell_accuracy = c_acc_correct/c_acc_total
    total_accuracy = acc_correct/total
    dim_accuracy = dim_correct/total

    print(f"ANALYZED {total} TASKS")
    print(f"CORRECT {acc_correct} TASKS")
    print(f"ACCURACY: {total_accuracy}")
    print(f"DIMENSION ACCURACY: {dim_accuracy}")
    print(f"CELL ACCURACY: {cell_accuracy}")
    print(all_correct)
    print(f"{format_percentage(dim_accuracy)} & {format_percentage(cell_accuracy)} & {acc_correct} & {format_percentage(total_accuracy)} \\\\")
