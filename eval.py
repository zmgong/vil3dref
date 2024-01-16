import numpy as np
import torchmetrics
import json
import pandas as pd

def cal_acc(pred, gt):
    correct_count = 0
    for i in range(len(pred)):
        if pred[i] == gt[i]:
            correct_count = correct_count + 1

    return correct_count*1.0/len(pred)


if __name__ == '__main__':
    with open('datasets/exprs_neurips22/gtlabelpcd_mix/sr3d/preds/val_outs.json') as json_file:
        pred_data = json.load(json_file)

    df = pd.read_csv('datasets/referit3d/annotations/bert_tokenized/sr3d.csv', index_col=False)
    gt_data = {}
    for index, row in df.iterrows():
        row_dict = dict(row)
        key = str(row_dict['item_id'])
        # number_of_zeros = 6-len(key)
        # key = "nr3d_" + "0" * number_of_zeros + key
        gt_data[key] = row_dict


    overall_pred_list = []
    overall_gt_list = []

    easy_pred_list = []
    easy_gt_list = []

    hard_pred_list = []
    hard_gt_list = []

    view_dep_pred_list = []
    view_dep_gt_list = []

    view_indep_pred_list = []
    view_indep_gt_list = []
    for item_id in pred_data:

        curr_pred_data = pred_data[item_id]
        curr_gt_data = gt_data[item_id]

        pred_idx = np.argmax(np.array(curr_pred_data['obj_logits']))

        pred_id = int(curr_pred_data['obj_ids'][pred_idx])
        gt_id = int(curr_gt_data['target_id'])

        # print("----------------------------------------------------------------------")
        # print(item_id)
        # print("predict id: " + str(pred_id))
        # print("target id: " + str(gt_id))
        # print(curr_pred_data)
        # print(curr_gt_data)
        # print("----------------------------------------------------------------------")
        # print()

        overall_pred_list.append(pred_id)
        overall_gt_list.append(gt_id)

        if curr_gt_data['is_easy']:
            easy_pred_list.append(pred_id)
            easy_gt_list.append(gt_id)
        else:
            hard_pred_list.append(pred_id)
            hard_gt_list.append(gt_id)
        if curr_gt_data['is_view_dep']:
            view_dep_pred_list.append(pred_id)
            view_dep_gt_list.append(gt_id)
        else:
            view_indep_pred_list.append(pred_id)
            view_indep_gt_list.append(gt_id)

    print(f'Overall_acc: {cal_acc(overall_pred_list, overall_gt_list)}')
    print(f'Easy_acc: {cal_acc(easy_pred_list, easy_gt_list)}')
    print(f'Hard_acc: {cal_acc(hard_pred_list, hard_gt_list)}')
    print(f'View_dep_acc: {cal_acc(view_dep_pred_list, view_dep_gt_list)}')
    print(f'View_indep_acc: {cal_acc(view_indep_pred_list, view_indep_gt_list)}')