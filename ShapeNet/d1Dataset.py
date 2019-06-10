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

h5_filepaths = [
    '_d1_a1.h5',
    '_d1_a2.h5',
    '_d1_a3.h5',
    '_d1_a4.h5',
    '_d1_a5.h5',
    '_d1_a6.h5',
    '_d1_a7.h5',
    '_d1_a8.h5',
    '_d1_a9.h5',
    '_d1_a11.h5',
    '_d1_a12.h5',
    '_d1_a13.h5',
    '_d1_a14.h5',
    '_d1_a16.h5',
]

CAT_NAME =  ["Airplane", 
            "Bag",     
            "Cap",     
            "Car",    
            "Chair",   
            "Earphone", 
            "Guitar",  
            "Knife",   
            "Lamp",    
            "Motorbike",
            "Mug",     
            "Pistol",  
            "Rocket",  
            "Table"]

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
        cat_ (nx2 array): List of tuples (category name, category folder) of the categories in the dataset.
        segClasses_ (dictionary of arrays): Each entry of the dictionary has a key equal to the name of the
                category and a list of part identifiers.
    """
    
    def __init__(self, train, batchSize, ptDropOut, allowedSamplings=[0], augment=False, seed=None):
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
            seed (int): Seed used to initialize the random number generator. If None is provided instead, the current
                time on the machine will be used to initialize the number generator.
        """
        
        # Store the parameters of the class.
        # Call the constructor of the parent class.
        super(d1DataSet,self).__init__(0, ptDropOut, pointFeatures=True, pointLabels=True, 
            pointNormals=True, useCategories=True, pointCategories=True, 
            batchSize=batchSize, allowedSamplings=allowedSamplings,
            augment=augment, augmentSmallRotations=True, seed=seed)

        self.h5_filepaths = [os.path.join('h5_d1', 'MCCNN' + ('_train' if train else '_test') + f) for f in h5_filepaths]

        self.records = None
        # original
        self.categories_ = None
        self.targets = None
        self.record_2_scene = None
        self.scenes = None
        self.record_2_cam_params = None
        self.record_2_hl = None

        self.record_interval = []
        self.scene_interval = []

        for cls_i, h5_filepath in enumerate(self.h5_filepaths):
            # store interval begin
            beg = 0 if self.records is None else len(self.records)
            scene_beg = 0 if self.scenes is None else len(self.scenes)
            # get data from file
            f = h5py.File(h5_filepath)
            tmp_records = f.get('record')[()]
            tmp_targets = f.get('target')[()]
            tmp_record_2_scene = f.get('record_2_scene')[()].astype(np.uint16)
            tmp_scenes = [s.reshape((-1, 3)) for s in f.get('scene')[()]]
            tmp_record_2_cam_params = f.get('record_2_cam_pos')[()]
            tmp_record_2_hl = f.get('recrod_2_highlight')[()]
            f.close()
            tmp_record_2_cls = np.full((len(tmp_records)), cls_i)

            # concat
            self.records = concat(self.records, [t.reshape((-1, 1)) for t in tmp_records])
            self.targets = concat(self.targets, [t.reshape((-1, 1)) for t in tmp_targets])
            tmp_record_2_scene += scene_beg
            self.record_2_scene = concat(self.record_2_scene, tmp_record_2_scene) 
            self.scenes = concat(self.scenes, tmp_scenes) 
            self.record_2_cam_params = concat(self.record_2_cam_params, tmp_record_2_cam_params)
            self.record_2_hl = concat(self.record_2_hl, tmp_record_2_hl)
            self.categories_ = concat(self.categories_, tmp_record_2_cls)

            # update interval with end
            self.record_interval.append((beg, len(self.records)))
            self.scene_interval.append((scene_beg, len(self.scenes)))

        if type(self.records) is list:
            self.records = np.array(self.records, dtype=np.object)
            self.targets = np.array(self.targets, dtype=np.object)
            self.scenes = np.array(self.scenes, dtype=np.object) 

        print('load records.shape:', self.records.shape,
            'targets.shape', self.targets.shape,
            'record_2_scene.shape:', self.record_2_scene.shape,
            'scenes.shape:', self.scenes.shape,
            'record_2_cam_params.shape:', self.record_2_cam_params.shape)

        # Get the categories and their associated part ids..
        self.catNames_ = [ CAT_NAME[cls_i] for cls_i, _ in enumerate(h5_filepaths)]

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

    def _load_model_(self, record_idx):
        scene_idx = self.record_2_scene[record_idx]
        pts = self.scenes[scene_idx]
        features = self.records[record_idx].astype(np.uint8)
        labels = self.targets[record_idx].astype(np.uint8)
        return  pts, None, features, labels

if __name__ == "__main__":
    dataset = d1DataSet(True, batchSize=32, ptDropOut=0.8, allowedSamplings=[0], augment=False)
    print(dataset.get_num_models(), dataset.get_categories())
    # dataset.start_iteration()
    numModelInBatch, accumPts, accumBatchIds, accumFeatures, accumLabels, accumCat, accumPaths, tick = dataset.get_next_batch(num_gpu=2)
    print(numModelInBatch)
    print('accumPts.shape', accumPts.shape)
    print('accumBatchIds.shape', accumBatchIds.shape)
    print('accumFeatures.shape', accumFeatures.shape)
    print('accumLabels.shape', accumLabels.shape)
    print('accumCat.shape', accumCat.shape)
    print(accumPaths)
    print(tick)

    dataset = d1DataSet(False, batchSize=32, ptDropOut=0.8, allowedSamplings=[0], augment=False)
    print(dataset.get_num_models(), dataset.get_categories())
    # dataset.start_iteration()
    numModelInBatch, accumPts, accumBatchIds, accumFeatures, accumLabels, accumCat, accumPaths, tick = dataset.get_next_batch(num_gpu=2)
    print(numModelInBatch)
    print('accumPts.shape', accumPts.shape)
    print('accumBatchIds.shape', accumBatchIds.shape)
    print('accumFeatures.shape', accumFeatures.shape)
    print('accumLabels.shape', accumLabels.shape)
    print('accumCat.shape', accumCat.shape)
    print(accumPaths)
    print(tick)