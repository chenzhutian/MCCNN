'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file d1DataSet.py

    \brief d1 dataset class.

    \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import os
import math
import time
import json
import numpy as np
import h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from Selection_Dataset import SelectionDataSet

h5_filepaths = []

def concat(main, sub):
    if type(main) is list:
        if type(sub) is not list:
            sub = [t for t in sub]
        main += sub
        return main
    else:
        return sub if main is None else np.concatenate((main, sub), axis=0)

class d1DataSet(SelectionDataSet):
    """d1 dataset.

    Attributes: 
        useNormalsAsFeatures_ (bool): Boolean that indicates if the normals will be used as the input features.
        cat_ (nx2 array): List of tuples (category name, category folder) of the categories in the dataset.
        segClasses_ (dictionary of arrays): Each entry of the dictionary has a key equal to the name of the
                category and a list of part identifiers.
    """
    
    def __init__(self, train, batchSize, ptDropOut, allowedSamplings=[0], augment=False, 
        useNormalsAsFeatures=False, seed=None):
        """Constructor.

        Args:
            train (bool): Boolean that indicates if this is the train or test dataset.
            batchSize (int): Size of the batch used.
            ptDropOut (float): Probability to keep a point during uniform sampling when all the points
                or only the first n number of points are selected.
            allowedSamplings (array of ints): Each element of the array determines an allowed sampling protocol
                that will be used to sample the different models. The implemented sampling protocols are:
                - 0: Uniform sampling
                - 1: Split sampling
                - 2: Gradient sampling
                - 3: Lambert sampling
                - 4: Occlusion sampling
            augment (bool): Boolean that indicates if data augmentation will be used in the models.
            useNormalsAsFeatures (bool): Boolean that indicates if the normals will be used as the input features.
            seed (int): Seed used to initialize the random number generator. If None is provided instead, the current
                time on the machine will be used to initialize the number generator.
        """
        
        # Store the parameters of the class.
        self.useNormalsAsFeatures_ = useNormalsAsFeatures

        # Call the constructor of the parent class.
        super(d1DataSet,self).__init__(0, ptDropOut, useNormalsAsFeatures, True, True,
            True, True, batchSize, allowedSamplings, 100000000, 0,
            augment, 1, True, False, [], [], seed)

        self.h5_filepaths = h5_filepaths
        self.records = None
        # original
        self.categories_ = None
        self.targets = None
        self.record_2_scene = None
        self.scenes = None
        self.record_2_cam_params = None
        self.record_2_hl = None

        for cls_i, h5_filepath in enumerate(h5_filepaths):
            # store interval begin
            beg = 0 if self.records is None else len(self.records)
            scene_beg = 0 if self.scenes is None else len(self.scenes)
            # get data from file
            f = h5py.File(h5_filepath)
            tmp_records = f.get('record')[()]
            tmp_targets = f.get('target')[()]
            tmp_record_2_scene = f.get('record_2_scene')[()].astype(np.uint16)
            tmp_scenes = f.get('scene')[()]
            tmp_record_2_cam_params = f.get('record_2_cam_pos')[()]
            tmp_record_2_hl = f.get('recrod_2_highlight')[()]
            f.close()
            tmp_record_2_cls = np.full((len(tmp_records)), cls_i)

            # concat
            self.records = concat(self.records, [t for t in tmp_records])
            self.targets = concat(self.targets, [t for t in tmp_targets])
            tmp_record_2_scene += scene_beg
            self.record_2_scene = concat(self.record_2_scene, tmp_record_2_scene) 
            self.scenes = concat(self.scenes, [s for s in tmp_scenes]) 
            self.record_2_cam_params = concat(self.record_2_cam_params, tmp_record_2_cam_params)
            self.record_2_hl = concat(self.record_2_hl, tmp_record_2_hl)
            self.categories_ = concat(self.categories_, tmp_record_2_cls)

            # update interval with end
            self.record_interval.append((beg, len(self.records)))
            self.scene_interval.append((scene_beg, len(self.scenes)))

        if type(self.records) is list:
            self.records = np.array(self.records, dtype=np.bool)
            self.targets = np.array(self.targets, dtype=np.bool)
            self.scenes = np.array(self.scenes, dtype=np.float32) 

        print('load records.shape:', self.records.shape,
            'targets.shape', self.targets.shape,
            'record_2_scene.shape:', self.record_2_scene.shape,
            'scenes.shape:', self.scenes.shape,
            'record_2_cam_params.shape:', self.record_2_cam_params.shape)

        # Get the categories and their associated part ids..
        self.catNames_ = [
            ["Airplane", "02691156"],
            ["Bag",     "02773838"],
            ["Cap",     "02954340"],
            ["Car",    "02958343"],
            ["Chair",   "03001627"],
            ["Earphone", "03261776"],
            ["Guitar",  "03467517"],
            ["Knife",   "03624134"],
            ["Lamp",    "03636649"],
            ["Motorbike", "03790512"],
            ["Mug",     "03797390"],
            ["Pistol",  "03948459"],
            ["Rocket",  "04099429"],
            ["Table",   "04379243"],
        ]

        # should be revised
        self.segClasses_ = {
            'Earphone': [16, 17, 18], 
            'Motorbike': [30, 31, 32, 33, 34, 35], 
            'Rocket': [41, 42, 43], 
            'Car': [8, 9, 10, 11], 
            'Laptop': [28, 29], 
            'Cap': [6, 7], 
            'Skateboard': [44, 45, 46], 
            'Mug': [36, 37], 
            'Guitar': [19, 20, 21], 
            'Bag': [4, 5], 
            'Lamp': [24, 25, 26, 27], 
            'Table': [47, 48, 49], 
            'Airplane': [0, 1, 2, 3], 
            'Pistol': [38, 39, 40], 
            'Chair': [12, 13, 14, 15], 
            'Knife': [22, 23]
        }

        # Since we do not know the size of the models in advance 
        # we initialize them to 0 and the first that will be loaded
        # this values will be update automatically.
        self.numPts_ = [len(self.scenes[scene_idx]) for scene_idx in self.record_2_scene]
        self.recordList_ = range(len(self.records))

    def get_categories(self):
        """Method to get the list of categories.
            
        Returns:
            pts (nx2 np.array string): List of tuples with the category name and the folder name.
        """
        return self.catNames_


    def get_categories_seg_parts(self):
        """Method to get the list of parts per category.
            
        Returns:
            pts (dict of array): Each entry of the dictionary has a key equal to the name of the
                category and a list of part identifiers.
        """
        return self.segClasses_

    def get_next_batch(self, num_gpu=1):
        numModelInBatch, accumPts, accumBatchIds, accumFeatures, accumLabels, accumCat, accumPaths = super().get_next_batch()
        tick = []

        model_per_gpu = math.ceil(numModelInBatch / num_gpu)
        for i in range(num_gpu):
            beg = i * model_per_gpu
            end = beg + model_per_gpu
            idx, _ = np.where((accumBatchIds < end) & (accumBatchIds >= beg))
            beg = min(idx)
            end = max(idx)
            tick.append((beg, end))

        return numModelInBatch, accumPts, accumBatchIds, accumFeatures, accumLabels, accumCat, accumPaths, tick

    def _load_model(self, record_idx):
        scene_idx = self.record_2_scene[record_idx]
        pts = self.scenes[scene_idx]
        features = self.records[record_idx]
        labels = self.targets[record_idx]

        return  pts, None, features, labels
