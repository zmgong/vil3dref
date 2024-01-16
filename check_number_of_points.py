import argparse
import json
import os

import numpy as np
import open3d as o3d
import pandas as pd
import torch
import tqdm
import matplotlib.pyplot as plt
import matplotlib

def get_bounding_box_with_specific_label_id(points, label_id, is_gt=True):
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


def create_custom_histogram(point_counts):
    print(max(point_counts))
    print(min(point_counts))

    ranges = [(0, 999), (1000, 1999), (2000, 2999), (3000, 3999), (4000, 4999), (5000, 5999), (6000, 6999), (7000, 7999), (8000, 8999), (9000, 9999), (10000, 19999), (20000, 29999), (30000, 39999), (40000, 49999), (50000, 59999), (60000, 69999), (70000, 79999), (80000, 89999), (90000, 99999), (100000, 109999), (110000, 119999), (120000, 129999), (130000, 139999), (140000, 149999), (150000, 159999)]
    categories = ['0, 999', '1000, 1999', '2000, 2999', '3000, 3999', '4000, 4999', '5000, 5999', '6000, 6999', '7000, 7999', '8000, 8999', '9000, 9999', '10000, 19999', '20000, 29999', '30000, 39999', '40000, 49999', '50000, 59999', '60000, 69999', '70000, 79999', '80000, 89999', '90000, 99999', '100000, 109999', '110000, 119999', '120000, 129999', '130000, 139999', '140000, 149999', '150000, 159999']

    category_counts = [0] * len(categories)

    for count in point_counts:
        for i, (start, end) in enumerate(ranges):
            if start <= count <= end:
                category_counts[i] += 1
                break

    plt.bar(categories, category_counts, edgecolor='black')
    plt.xticks(rotation=30)
    plt.yscale('log')
    plt.xlabel('Point Count Range')
    plt.ylabel('Number of Objects')

    plt.title('Point Count Distribution for sr3d')
    plt.show()

def generate_histogram(count_dict, des='Average'):
    bins = list(count_dict.keys())
    counts = list(count_dict.values())

    plt.bar(bins, counts)
    plt.xticks(rotation=30)
    plt.xlabel('Classes')
    plt.ylabel(des + ' number of points')
    plt.title('Point Count Distribution over Classes sr3d')
    plt.show()

def draw_box_plot(count_dict):
    data_values = list(count_dict.values())
    print(len(list(count_dict.keys())))
    plt.boxplot(data_values, labels=list(count_dict.keys()))
    plt.title('Boxplot of Int Lists')
    plt.xlabel('Groups')
    plt.ylabel('Values')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_folder', type=str, default='datasets/referit3d/annotations/bert_tokenized')
    parser.add_argument('--file_name', type=str, default='sr3d')
    parser.add_argument('--output_dir', type=str, default='datasets/referit3d/annotations/bert_tokenized/with_point_count')
    parser.add_argument('--scannet_dir', type=str, default='datasets/referit3d/scan_data')
    parser.add_argument('--scannet_labels', type=str, default='datasets/referit3d/annotations/meta_data/scannetv2-labels.combined.tsv')
    args = parser.parse_args()
    path_to_pth_files = os.path.join(args.scannet_dir, "pcd_with_global_alignment")
    gt_data = None
    if not os.path.exists(os.path.join(args.output_dir, args.file_name) + '.json'):
        os.makedirs(args.output_dir, exist_ok=True)
        gt_df = pd.read_csv(os.path.join(args.ann_folder, args.file_name) + '.csv', index_col=False)
        label_map_df = pd.read_csv(args.scannet_labels, index_col=False, sep='\t')
        gt_data = gt_df.set_index('item_id').to_dict(orient='index')

        map_dict = label_map_df[['raw_category', 'nyu40class']].set_index('raw_category').to_dict()['nyu40class']
        count_list = []
        count_dict = {}
        for curr_item_id in tqdm.tqdm(gt_data.keys()):
            curr_gt_data = gt_data[curr_item_id]
            curr_gt_data['instance_type'] = map_dict[curr_gt_data['instance_type']]

            gt_id = int(curr_gt_data['target_id'])

            pcd_data = torch.load(os.path.join(path_to_pth_files, curr_gt_data['scan_id'] + '.pth'))
            points, colors, labels = pcd_data[0], pcd_data[1] / 255.0, pcd_data[-1]
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            gt_bounding_box = get_bounding_box_with_specific_label_id(points, gt_id, is_gt=True)

            gt_data[curr_item_id]['number_of_points'] = len(gt_bounding_box.get_point_indices_within_bounding_box(point_cloud.points))

        with open(os.path.join(args.output_dir, args.file_name) + '.json', 'w') as json_file:
            json.dump(gt_data, json_file, indent=4)
    with open(os.path.join(args.output_dir, args.file_name) + '.json', 'r') as json_file:
        gt_data = json.load(json_file)

    count_list = []
    count_dict = {}

    for curr_item_id in gt_data.keys():
        curr_gt_data = gt_data[curr_item_id]
        curr_label = curr_gt_data['instance_type']
        curr_point_count = curr_gt_data['number_of_points']

        if curr_label not in count_dict.keys():
            count_dict[curr_label] = []

        count_list.append(curr_point_count)
        count_dict[curr_label].append(curr_point_count)

    font = {
            'size': 15}

    matplotlib.rc('font', **font)
    create_custom_histogram(count_list)

    # avg_dict = {}
    # for curr_label in count_dict.keys():
    #     avg_dict[curr_label] = sum(count_dict[curr_label])*1.0/len(count_dict[curr_label])
    # avg_dict = dict(sorted(avg_dict.items(), key=lambda item: item[1], reverse=True))
    # generate_histogram(avg_dict)
    #
    # max_dict = {}
    # for curr_label in count_dict.keys():
    #     max_dict[curr_label] = max(count_dict[curr_label])
    # max_dict = dict(sorted(max_dict.items(), key=lambda item: item[1], reverse=True))
    # generate_histogram(max_dict, des='Max')
    #
    # min_dict = {}
    # for curr_label in count_dict.keys():
    #     min_dict[curr_label] = min(count_dict[curr_label])
    # min_dict = dict(sorted(min_dict.items(), key=lambda item: item[1], reverse=True))
    # generate_histogram(min_dict, des='Min')