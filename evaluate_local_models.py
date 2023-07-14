#!/usr/bin/env python
import json
import os

RES_START = "Out:"


def find_result(result, exp_result):
    exp_result = exp_result.removeprefix(RES_START).strip()
    min_exp_result = exp_result.replace(' ', '')
    min_exp_chars = set(min_exp_result)
    candidates = []
    for r in result.split(RES_START):
        # Result has to be in the same line, otherwise ignore
        r = r.strip().split('\n')[0]
        min_a = r.replace(' ', '')
        common_pref = os.path.commonprefix([min_exp_result, min_a])
        # Simple case, we have a perfect match
        if common_pref == min_exp_result:
            return exp_result
        # Too little in common
        if len(common_pref) <= 1:
            continue
        # Remember really common results
        if len(common_pref) >= 0.9*len(min_exp_result):
            candidates.append((len(common_pref), r))
            continue
        # Fallback case: use the ratio of similar chars times strlen as metric
        min_a_chars = set(min_a)
        intersec = min_a_chars.intersection(min_exp_chars)
        candidates.append((len(intersec)/len(min_exp_chars) * len(r), r))

    # Find best candidate
    max_l, max_r = 0, None
    for (l, r) in candidates:
        if l > max_l:
            max_l, max_r = l, r

    return max_r


def interpret_result_as_arr(result):
    res = []
    if not result:
        return res
    if ']' in result:
        lines = result.split(']')
    else:
        lines = result.split(';')
    for line in lines:
        res_inner = []
        for c in line.strip():
            if c.isdigit():
                res_inner.append(int(c))
        res.append(res_inner)
    return res


def evaluate_result(result, exp_result):
    result_arr = interpret_result_as_arr(result)
    exp_result_arr = interpret_result_as_arr(exp_result)

    correct = str(result_arr) == str(exp_result_arr)
    dim_correct = len(result_arr) == len(exp_result_arr) and all([
        len(a) == len(b)
        for (a, b) in zip(result_arr, exp_result_arr)
    ])
    elems_correct = sum([
        sum([a_ == b_ for (a_, b_) in zip(a, b)])
        for (a, b) in zip(result_arr, exp_result_arr)
    ]) if dim_correct else 0
    elems_total = sum([
        len(a) for a in exp_result_arr
    ])

    return {
        'correct': correct,
        'dim_correct': dim_correct,
        'elems_correct': elems_correct,
        'elems_total': elems_total,
    }


def evaluate_json(file, show_each_task=True):
    with open(file, "r") as data:
        try:
            d = json.load(data)
        except:
            print("Failed reading file " + file)
            return

    if not d:
        print("Failed reading file " + file)
        return

    overall_correct, overall_dim_correct = 0, 0
    overall_no_correct_elems, overall_no_total_elems = 0, 0
    for elem in d:
        taskId, exp_result = elem["taskId"], elem["exp_result"]

        result = find_result(elem["result"], exp_result)
        eval = evaluate_result(result, exp_result)

        overall_correct += eval['correct']
        overall_dim_correct += eval['dim_correct']
        if eval['dim_correct']:
            overall_no_correct_elems += eval['elems_correct']
            overall_no_total_elems += eval['elems_total']

        if show_each_task:
            print(f"task: {taskId}")
            print(f"expected result:  {exp_result}")
            print(f"generated result: {result}")
            print("task score:")
            print(f"\t overall correctness: {eval['correct']}")
            print(f"\t dimens. correctness: {eval['dim_correct']}")
            print(f"\t no correct elems:    {eval['elems_correct']}")
            print(f"\t no total elems:      {eval['elems_total']}")
            print()

    overall_correct /= len(d)
    overall_dim_correct /= len(d)
    overall_correct_elems = overall_no_correct_elems / max(overall_no_total_elems, 1)

    name = ' '.join(os.path.dirname(file)
                    .replace('results/', '')
                    .split('shARCathon')[1]
                    .split('__main__.py'))
    print("--------------------------", end=' ')
    print(f"{name}", end=' ')
    print("--------------------------")
    print(f"overall correctness (%): {round(overall_correct * 100, 4)}")
    print(f"dimens. correctness (%): {round(overall_dim_correct * 100, 4)}")
    print(f"correct elems (%):       {round(overall_correct_elems * 100, 4)}")
    print(f"correct elems (abs):     {overall_no_correct_elems}")


def main():
    for d in os.listdir('./results/'):
        for f in os.listdir(f'./results/{d}'):
            path = os.path.join('./results/', d, f)
            evaluate_json(path, False)


if __name__ == "__main__":
    main()
    # evaluate_json(
    #     "./results/workspaceshARCathongptneo__main__.pybasicapply-patternsnormal/1689295541.9080024.json")
