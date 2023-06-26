import os
import json

import config
import utils

def compute_accuracy(pred_matrix, gt_matrix):
    correct = 0
    total = 0
    for pred_row, gt_row in zip(pred_matrix, gt_matrix):
            for pred_cell, gt_cell in zip(pred_row, gt_row):
                total += 1
                if pred_cell == gt_cell:
                    correct += 1
    # Calculate and return the accuracy
    accuracy = correct / total if total else 0
    return accuracy


def compute_full_accuracy(predicted, ground_truth):
    correct = 0
    total = 0
    correct_dimensions = 0
    # Iterate over each pair of matrices
    for pred_matrix, gt_matrix in zip(predicted, ground_truth):
        # Ensure both matrices have the same dimensions
        if len(pred_matrix) == len(gt_matrix): #and all(len(pred_row) == len(gt_row) for pred_row, gt_row in zip(pred_matrix, gt_matrix)):
            correct_dimensions += 1
            # Iterate over each cell in the matrices
            for pred_row, gt_row in zip(pred_matrix, gt_matrix):
                for pred_cell, gt_cell in zip(pred_row, gt_row):
                    total += 1
                    if pred_cell == gt_cell:
                        correct += 1
    # Calculate and return the accuracy
    
    return correct, total, correct_dimensions


def main():
    train_data = utils.load_train_json(config.TRAIN_PATH)
    print(train_data)
    results_data = utils.load_results_json(config.OUT_PRED_PATH)
    #print(results_data)
    pred_list = results_data.values()
    train_list = train_data.values()
    #compute_accuracy(pred_list, train_list)
    for entry in train_data:
        print("Entry ", entry)
        print("GT ", train_data[entry])
        print("Prediction ", results_data[entry])
        predicted = results_data[entry]
        accuracy = compute_accuracy(predicted, train_data[entry])
        print("ACCURACY ", accuracy)
    correct, total, corret_dimensions = compute_full_accuracy(pred_list, train_list)
    print("Correct ", correct)
    print("Total ", total)
    print("ACCURACY ", correct/total)
    print("Correct dimensions ", corret_dimensions)
    #print(train_data)
    #print(results_data)
    

if __name__=="__main__":
    main()