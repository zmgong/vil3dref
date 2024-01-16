import argparse
import json
import os

import numpy as np
import open3d as o3d
import pandas as pd
import torch
from plyfile import PlyData

def get_instance_labels(scan_id, scan_dir):
    with open(os.path.join(scan_dir, scan_id, '%s_vh_clean_2.ply' % (scan_id)), 'rb') as f:
        plydata = PlyData.read(f)  # elements: vertex, face
    points = np.array([list(x) for x in plydata.elements[0]])  # [[x, y, z, r, g, b, alpha]]
    coords = np.ascontiguousarray(points[:, :3])
    colors = np.ascontiguousarray(points[:, 3:6])

    with open(os.path.join(scan_dir, scan_id, '%s_vh_clean_2.labels.ply' % (scan_id)), 'rb') as f:
        plydata = PlyData.read(f)
    sem_labels = np.array(plydata.elements[0]['label']).astype(np.longlong)
    assert len(coords) == len(colors) == len(sem_labels)
    # Map each point to segment id
    with open(os.path.join(scan_dir, scan_id, '%s_vh_clean_2.0.010000.segs.json' % (scan_id)), 'r') as f:
        d = json.load(f)
    seg = d['segIndices']
    segid_to_pointid = {}
    for i, segid in enumerate(seg):
        segid_to_pointid.setdefault(segid, [])
        segid_to_pointid[segid].append(i)
    # Map object to segments
    instance_class_labels = []
    instance_segids = []
    with open(os.path.join(scan_dir, scan_id, '%s.aggregation.json' % (scan_id)), 'r') as f:
        d = json.load(f)
    for i, x in enumerate(d['segGroups']):
        assert x['id'] == x['objectId'] == i
        instance_class_labels.append(x['label'])
        instance_segids.append(x['segments'])
    instance_labels = np.ones(sem_labels.shape[0], dtype=np.longlong) * -100
    for i, segids in enumerate(instance_segids):
        pointids = []
        for segid in segids:
            pointids += segid_to_pointid[segid]
        if np.sum(instance_labels[pointids] != -100) > 0:
            # scene0217_00 contains some overlapped instances
            print(scan_id, i, np.sum(instance_labels[pointids] != -100), len(pointids))
        else:
            instance_labels[pointids] = i
            assert len(np.unique(sem_labels[pointids])) == 1, 'points of each instance should have the same label'
    return instance_labels

def get_bbox(predicted_mask, points):
    x_min = None
    y_min = None
    z_min = None
    x_max = None
    y_max = None
    z_max = None
    for vertexIndex, xyz in enumerate(points):
        if predicted_mask[vertexIndex] == True:
            if x_min is None or xyz[0] < x_min:
                x_min = xyz[0]
            if y_min is None or xyz[1] < y_min:
                y_min = xyz[1]
            if z_min is None or xyz[2] < z_min:
                z_min = xyz[2]
            if x_max is None or xyz[0] > x_max:
                x_max = xyz[0]
            if y_max is None or xyz[1] > y_max:
                y_max = xyz[1]
            if z_max is None or xyz[2] > z_max:
                z_max = xyz[2]
    return x_min, x_max, y_min, y_max, z_min, z_max


def check_gt_pred_and_gt_pred(curr_pred_data, curr_gt_data):
    pred_idx = np.argmax(np.array(curr_pred_data['obj_logits']))

    pred_id = int(curr_pred_data['obj_ids'][pred_idx])
    gt_id = int(curr_gt_data['target_id'])

    return pred_id, gt_id


def get_bounding_box_with_specific_label_id(points, labels, label_id, is_gt=True):
    gt_indices_of_points = np.where(labels == label_id)
    gt_points = points[gt_indices_of_points]

    min_bound = np.min(gt_points, axis=0)
    max_bound = np.max(gt_points, axis=0)

    if is_gt:
        color = np.array([1.0, 0.0, 0.0])
    else:
        color = np.array([0.0, 1.0, 0.0])

    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    bounding_box.color = color

    return bounding_box


def read_axis_align_matrix(file_path):
    axis_align_matrix = None
    with open(file_path, "r") as f:
        for line in f:
            line_content = line.strip()
            if 'axisAlignment' in line_content:
                axis_align_matrix = [float(x) for x in line_content.strip('axisAlignment = ').split(' ')]
                axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
                break
    return axis_align_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_json', type=str,
                        default='datasets/exprs_neurips22/gtlabelpcd_mix/nr3d/preds/val_outs.json')
    parser.add_argument('--gt_csv', type=str, default='datasets/referit3d/annotations/bert_tokenized/nr3d.csv')
    parser.add_argument('--scannet_dir', type=str, default='/datasets/released/scannet/public/v2/scans')
    args = parser.parse_args()


    with open(args.predict_json) as json_file:
        pred_data = json.load(json_file)

    gt_df = pd.read_csv(args.gt_csv, index_col=False)
    gt_data = gt_df.set_index('item_id').to_dict(orient='index')

    correct_count = 0
    wrong_count = 0
    for curr_item_id in pred_data.keys():
        curr_pred_data = pred_data[curr_item_id]
        curr_gt_data = gt_data[curr_item_id]
        pred_id, gt_id = check_gt_pred_and_gt_pred(curr_pred_data, curr_gt_data)
        if pred_id == gt_id:
            continue
        else:

            ply_file = os.path.join(args.scannet_dir, curr_gt_data["scan_id"], f'{curr_gt_data["scan_id"]}_vh_clean_2.ply')
            mesh = o3d.io.read_triangle_mesh(ply_file)
            alignment_file = os.path.join(args.scannet_dir, curr_gt_data["scan_id"], f'{curr_gt_data["scan_id"]}.txt')
            align_matrix = read_axis_align_matrix(alignment_file)
            mesh.transform(align_matrix)

            points = np.array(mesh.vertices)

            labels = get_instance_labels(curr_gt_data["scan_id"], args.scannet_dir)

            gt_bounding_box = get_bounding_box_with_specific_label_id(points, labels, gt_id, is_gt=True)
            pred_bounding_box = get_bounding_box_with_specific_label_id(points, labels, pred_id, is_gt=False)

            o3d.visualization.draw_geometries([mesh, gt_bounding_box, pred_bounding_box])

        break
