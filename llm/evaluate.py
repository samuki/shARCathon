# evaluate language model performance
import json
import os
import sys
sys.path.append('../')
import gpt_utils
import config




def naive_postprocessing(prediction):
    return prediction.split(':')[-1].replace("\n", "").strip()

# Path to the folders
#model_name = "gpt-3.5-turbo-16k"
model_name = "gpt-4"
replace_comma = "replace_comma"
#replace_comma = "no_replace_comma"
ground_truth_folder = "../data/evaluation_small"
#predictions_base_folder = "../results/better_prompt_evaluation_small"
predictions_base_folder = "../results/evaluation_small"
model_predictions_folder = f"{predictions_base_folder}/{replace_comma}/{model_name}"

# Get the list of task files
task_files = [f for f in os.listdir(ground_truth_folder) if f.endswith('.json')]
correct = 0
total = 0
all_correct = []
for task_file in task_files:
    # Open the corresponding model prediction file
    prediction_file = task_file.replace('.json', '_out.json')
    if os.path.exists(os.path.join(model_predictions_folder, prediction_file)):
        with open(os.path.join(model_predictions_folder, prediction_file), 'r') as pred_file:
            prediction = json.load(pred_file)
            
        # Open the ground truth file
        with open(os.path.join(ground_truth_folder, task_file), 'r') as gt_file:
            ground_truth = json.load(gt_file)

        print(f'Task {task_file}:')
        ground_truth_out = str(ground_truth['test'][0]['output'])
        ground_truth_out = gpt_utils.preprocess_representation(ground_truth_out)
        print(f'Ground truth: {ground_truth_out}')
        prediction_out = prediction['output']
        finish_reason = prediction_out["choices"][0]['finish_reason']
        result = prediction_out["choices"][0]["message"]["content"]
        postprocessed_result = naive_postprocessing(result)
        print(f'Finish reason: {finish_reason}')
        print("Prediction: ", postprocessed_result)
        #print("Prediction: ", result)
        # Check if the ground truth 'output' is equal to the model 'output'
        if ground_truth_out == postprocessed_result:
            print(f'Task {task_file}: Prediction is correct.')
            correct += 1
            all_correct.append(task_file)
        else:
            print(f'Task {task_file}: Prediction is incorrect.')
        total += 1
    else:
        print(f'Task {task_file}: No prediction file found.')
    print("\n")
print(f"ANALYZED {total} TASKS")
print(f"CORRECT {correct} TASKS")
print(f"ACCURACY: {correct/total}")
print(all_correct)