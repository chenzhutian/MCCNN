'''Prepare the h5 dataset for network trainning
'''
import os
from os import path
import re
import json

from pymongo import MongoClient
import numpy as np
import h5py

PTX_DIR = path.join('..', '2019-scivis-3dselection-label-tool', 'datasets', 'converted_d1')
OUTPUT_DIR = path.join('.', 'h5_d1')
if not path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
CONVERT_AREA_COUNT = 16
mongo_url='mongodb://127.0.0.1:27016'
db_name='pcSelection5'
collection_name='records'
FILTER_DATA = {
    'd1_a16_': [3]
}

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def convert_mongo_to_numpy(dataset_prefix):
    ## 0. connect db
    client = MongoClient(mongo_url)
    db = client[db_name]
    Records = db[collection_name]
    dataset_prefix_reg = re.compile('^' + dataset_prefix)

    features = []
    labels = []
    record_2_scene = []
    scenes = []
    record_2_cam_pos = []
    recrod_2_highlight = []

    # 1. find all records belong to d1
    records_by_room = Records.aggregate([
        { '$match': { 'pcId': { '$regex': dataset_prefix_reg }}},
        { '$project': { 'pcId': 1, '_id': 0 }},
        { '$group' : { '_id' : "$pcId"} }
    ], allowDiskUse=True)

    target_count = 0
    acc_count = 0
    # full attrs: x,y,z, isIn, isTarget
    EXPAND_DIM = 2 # isIn, isTarget

    # 2. construct the data of records in different room
    for item in records_by_room:
        pcId = item.get('_id')

        records = Records.aggregate([
            { '$match': { 'pcId': pcId, 'valid': True }},
            { '$project': { 'pcId': 1, 'points': 1, 'highlight': 1, 'cameraParams': 1, '_id': 0 }},
        ], allowDiskUse=True)
        records_count = Records.count({ 'pcId': pcId })

        points_file_path = path.join(PTX_DIR, pcId + '.txt')
        # check points file exist
        if not path.isfile(points_file_path):
            print("file " + points_file_path + " not exist")
            continue

        # open the points file of the pcId
        with open(path.join(PTX_DIR, pcId + '.txt')) as points_file:
            # 0. load and initialize data
            points_xyz_label_idx = np.array([l.strip().split(' ') for l in points_file ], dtype=np.float32)
            # 1. assign x, y, z
            basic_points = points_xyz_label_idx[:, :3]
            basic_label = points_xyz_label_idx[:, 3]
            current_points_count = len(basic_points)
            # save to scenes
            scene_id = len(scenes)
            scenes.append(basic_points)

            # process records
            for i, record in enumerate(records):
                try:
                    highlight = int(record.get('highlight'))
                    # check filter
                    filter_highlight = FILTER_DATA.get(dataset_prefix, [])
                    if highlight in filter_highlight:
                        continue

                    points = np.zeros((current_points_count, EXPAND_DIM), dtype=np.bool) # basic_points.copy()
                    # 2. assign isIn
                    idx_of_in = np.array(record.get('points'))
                    # how to extend the idx_of_in
                    points[idx_of_in, 0] = True
                    # 3. assign isTarget
                    idx_of_highlight = basic_label == highlight
                    points[idx_of_highlight, 1] = True
                    # 4. append data
                    features.append(points[:, 0])
                    labels.append(points[:, 1])
                    # 5. save camera position for dx, dy, dz
                    cam_params = record.get('cameraParams')
                    cam_pos = cam_params.get('position')
                    cam_rot = cam_params.get('rotation')
                    if cam_rot is None or cam_params.get('worldInverseMatrix') is None or cam_params.get('projectionMatrix') is None:
                        print('Attention, the cam_rot is None of ' + record.get('pcId'))
                    cam_data = np.concatenate((
                        [cam_pos.get('x'), cam_pos.get('y'), cam_pos.get('z'),
                        cam_rot.get('x'), cam_rot.get('y'), cam_rot.get('z')],
                        cam_params.get('worldInverseMatrix'),
                        cam_params.get('projectionMatrix')
                    )).astype(np.float32)
                    record_2_cam_pos.append(cam_data)
                    # 6. save x, y, z
                    record_2_scene.append(scene_id)
                    # 6.5 save highlight
                    recrod_2_highlight.append(highlight)
                    # 7. cal accuracy
                    target_count += points[(points[:, 0] == 1) + (points[:, 1] == 1)].shape[0]
                    acc_count    += points[(points[:, 0] == 1) * (points[:, 1] == 1)].shape[0]
                    # 8. print progress
                    
                except Exception as e:
                    print('On one of ' + pcId)
                    print(e)
        # printProgressBar(i + 1, records_count, prefix = pcId+':', suffix = 'Complete', length = 50)

    features = np.array(features, dtype=np.bool)
    labels = np.array(labels, dtype=np.bool)
    record_2_scene = np.array(record_2_scene, dtype=np.uint16)
    scenes = np.array(scenes, dtype=np.float32)
    record_2_cam_pos = np.array(record_2_cam_pos, dtype=np.float32)
    recrod_2_highlight = np.array(recrod_2_highlight, dtype=np.uint8)

    print(features.shape, labels.shape, record_2_scene.shape, scenes.shape, record_2_cam_pos.shape)
    if features.shape[0] == 0:
        return

    if target_count == 0:
        print('No target in ' + dataset_prefix)
    else:
        print('Average Accuracy of ' + dataset_prefix + ':', float(acc_count) / float(target_count))
    return features, labels, record_2_scene, scenes, record_2_cam_pos, recrod_2_highlight

# Write numpy array data and label to h5_filename
def save_h5(h5_filename, record, target, record_2_scene, scene, record_2_cam_pos, recrod_2_highlight):
    if path.isfile(h5_filename):
        os.remove(h5_filename)
    
    unit_factor = 1024 * 1024 * 1024
    print("record: %f gb" % (record.size * record.itemsize / unit_factor))
    print("target: %f gb" % (target.size * target.itemsize / unit_factor))
    print("record_2_scene: %f gb" % (record_2_scene.size * record_2_scene.itemsize / unit_factor))
    print("scene: %f gb" % (scene.size * scene.itemsize / unit_factor))
    print("record_2_cam_pos: %f gb" % (record_2_cam_pos.size * record_2_cam_pos.itemsize / unit_factor))
    print("recrod_2_highlight: %f gb" % (recrod_2_highlight.size * recrod_2_highlight.itemsize / unit_factor))

    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset('record', data=record,               
            compression='gzip', compression_opts=4, dtype='bool')
    h5_fout.create_dataset('target', data=target,
            compression='gzip', compression_opts=4, dtype='bool')
    h5_fout.create_dataset('record_2_scene', data=record_2_scene,
            compression='gzip', compression_opts=4, dtype='uint16')
    h5_fout.create_dataset('scene', data=scene,
            compression='gzip', compression_opts=4, dtype='float32')
    h5_fout.create_dataset('record_2_cam_pos', data=record_2_cam_pos,
            compression='gzip', compression_opts=4, dtype='float32')
    h5_fout.create_dataset('recrod_2_highlight', data=recrod_2_highlight,
            compression='gzip', compression_opts=4, dtype='uint8')
    h5_fout.close()

def check():
    for i in range(CONVERT_AREA_COUNT):
        aId = i + 1
        dataset_prefix = 'd1_a' + str(aId)
        room_files = [f for f in os.listdir(PTX_DIR) if dataset_prefix in f ]
        if len(room_files) == 0:
            continue
        point_counts = []
        highlight_counts = []
        for f in room_files:
            with open(path.join(PTX_DIR, f)) as points_file:
                points_xyz_label_idx = np.array([l.strip().split(' ') for l in points_file ], dtype=np.float32)
                # 1. assign x, y, z
                basic_points = points_xyz_label_idx[:, :3]
                current_points_count = len(basic_points)
                point_counts.append(current_points_count)
                if current_points_count < 2048:
                    print(f, ' less than 2048 by ', 2048 - current_points_count)
                idx_of_highlight = np.sum(points_xyz_label_idx[:, 3] == 3)
                if idx_of_highlight != 0:
                    highlight_counts.append(idx_of_highlight)
                    print('not 3 count ', current_points_count - idx_of_highlight)
        mean_counts = sum(point_counts) / len(point_counts)
        mean_highlight_counts = sum(highlight_counts) / len(highlight_counts)
        print(sorted(highlight_counts))
        print('aId ', aId, ' mean_counts ', mean_counts, 'mean_highlight_counts ', mean_highlight_counts, ' #', len(highlight_counts))

if __name__ == "__main__":
    for i in range(CONVERT_AREA_COUNT):
        aId = i + 1
        dataset_prefix = 'd1_a' + str(aId) + '_'
        results = convert_mongo_to_numpy(dataset_prefix)
        print('finish ', aId)
        if results is not None:
            record, target, record_2_scene, scene, record_2_cam_pos, recrod_2_highlight = results
            save_h5(path.join(OUTPUT_DIR, 'MCCNN_' + dataset_prefix[:-1] +'.h5'), record, target, record_2_scene, scene, record_2_cam_pos, recrod_2_highlight)
        