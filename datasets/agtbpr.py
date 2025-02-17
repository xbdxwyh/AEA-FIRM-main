# encoding: utf-8
"""
@author:  huynguyen792
@contact: nguyet91@qut.edu.au
"""

import glob
import re
import mat4py
import pandas as pd
import torch
import logging

import json

import os.path as osp

from prettytable import PrettyTable


class AG_ReID(object):
    logger = logging.getLogger("IRRA.dataset")
    dataset_dir = 'AG-ReID'

    def __init__(self, root='./data',
                 verbose=True, name="AGTBPR",text="aerial", **kwargs):
        super(AG_ReID, self).__init__()
        camera = 'a'
        self.text = text
        self.flag=0
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')

        self.query_dir = [
            osp.join(self.dataset_dir, 'query_all_c0'),
            osp.join(self.dataset_dir, 'bounding_box_test_all_c3')
        ]
        self.gallery_dir = [
            osp.join(self.dataset_dir, 'query_all_c3'),
            osp.join(self.dataset_dir, 'bounding_box_test_all_c0')
        ]
        # self.query_dir = osp.join(self.dataset_dir, 'query_all_c3')
        # self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test_all_c0')

        self.qut_attribute_path = osp.join(self.dataset_dir, 'qut_attribute_v4_88_attributes.mat')
        self.attribute_dict_all = self.generate_attribute_dict(self.qut_attribute_path, "qut_attribute")

        with open(osp.join(self.dataset_dir,"agtbpr_text.json"),"r") as f:
            data = json.load(f)

        self.text_captions = {list(d.keys())[0]:list(d.values())[0] for d in data}

        self._check_before_run()

        self.train, self.train_id_container = self._process_dir(self.train_dir, is_train=True, camera=camera)
        self.test, self.test_id_container = self._process_dir(self.query_dir, is_train=False, camera=camera)
        self.val, self.val_id_container = self._process_dir(self.gallery_dir, is_train=False, camera=camera)

        if verbose:
            self.logger.info("=> AGTBPR Images and Captions are loaded")
            self.show_dataset_info()

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = \
            len(self.train_id_container), len(self.train), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = \
            len(self.test_id_container), len(self.test['captions']), len(self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = \
            len(self.val_id_container), len(self.val['captions']), len(self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info("\n"+str(table))

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

        if not osp.exists(self.query_dir[0]) or not osp.exists(self.query_dir[1]):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir[0]) or not osp.exists(self.gallery_dir[1]):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, is_train=True, camera="a"):
        cid_range = [0] # 0 means the pic is getted by Aerials; 3 means Cameras.
 
        pid_list = []
        pid_container = set()    
        image_id = 0
        dataset = []
        image_pids = []
        image_paths = []
        caption_pids = []
        captions = []
        camera_ids = []
        gnd_img_paths = []
        if is_train:
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        else:
            img_paths = glob.glob(osp.join(dir_path[0], '*.jpg')) + glob.glob(osp.join(dir_path[1], '*.jpg'))
        
        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')

        # rerange pids
        for img_path in img_paths:
            fname = osp.split(img_path)[-1]
            pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
            pid = int(pid_part1 + pid_part2 + pid_part3)
            pid_list.append(pid)

        # set Aerial-camera pair
        pid_set = set(pid_list)
        pid_key = {pid:key for key,pid in enumerate(pid_set)}
        pid_dict = {key:[] for key in range(len(pid_key))}
        pid_dict_ptr = {key:0 for key in range(len(pid_key))}
        for img_path in img_paths:
            fname = osp.split(img_path)[-1]
            pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
            pid = int(pid_part1 + pid_part2 + pid_part3)
            pid = pid_key[pid]
            camid, frameid = pattern_camid.search(fname).groups()
            camid = int(camid)
            # purn the Camera images
            if camid not in cid_range:
                pid_dict[pid].append(img_path)

        # process data
        for img_path in img_paths:
            fname = osp.split(img_path)[-1]
            camid, frameid = pattern_camid.search(fname).groups()
            camid = int(camid)
            if camid not in cid_range:
                continue

            pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
            pid = int(pid_part1 + pid_part2 + pid_part3)
            pid = pid_key[pid]
            
            pid_container.add(pid)

            gnd_img_name = pid_dict[pid][pid_dict_ptr[pid]]
            pid_dict_ptr[pid] = (pid_dict_ptr[pid]+1) % len(pid_dict[pid])

            if self.text=="aerial":
                key = img_path
            elif self.text=="ground":
                key = gnd_img_name
            elif self.text=="half":
                key = img_path if self.flag else gnd_img_name
                self.flag = (self.flag+1) % 2
            else:
                raise NotImplementedError("no supportted text type{}".format(self.text))

            dir_path_tmp, fname_path_tmp = osp.split(key)
            # dir_path_tmp, fname_path_tmp = osp.split(img_path)
            fname_path_tmp = osp.splitext(fname_path_tmp)[0]
            _, dir_path_tmp = osp.split(dir_path_tmp)
            key = dir_path_tmp+"_"+fname_path_tmp

            if is_train:
                dataset.append((pid, image_id, img_path, gnd_img_name, self.text_captions[key]))
                image_id += 1
            else:
                image_pids.append(pid)
                image_paths.append(img_path)
                caption_pids.append(pid)
                gnd_img_paths.append(gnd_img_name)
                captions.append(self.text_captions[key])

        if not is_train:  
            dataset = {
                "image_pids": image_pids,
                "img_paths": image_paths,
                "pair_img_paths": gnd_img_paths,
                "caption_pids": caption_pids,
                "captions": captions,
                #"camera_ids": camera_ids
            }
        
        return dataset,pid_container

    def generate_attribute_dict(self, dir_path: str, dataset: str):

        mat_attribute_train = mat4py.loadmat(dir_path)[dataset]["train"]
        mat_attribute_train = pd.DataFrame(mat_attribute_train, index=mat_attribute_train['image_index']).astype(int)

        mat_attribute_test = mat4py.loadmat(dir_path)[dataset]["test"]
        mat_attribute_test = pd.DataFrame(mat_attribute_test, index=mat_attribute_test['image_index']).astype(int)

        mat_attribute = mat_attribute_train.add(mat_attribute_test, fill_value=0)
        mat_attribute = mat_attribute.drop(['image_index'], axis=1)

        self.key_attribute = list(mat_attribute.keys())

        h, w = mat_attribute.shape
        dict_attribute = dict()

        for i in range(h):
            row = mat_attribute.iloc[i:i + 1, :].values.reshape(-1)
            dict_attribute[str(int(mat_attribute.index[i]))] = torch.tensor(row[0:].astype(int)) * 2 - 3

        return dict_attribute

    def name_of_attribute(self):
        if self.key_attribute:
            print(self.key_attribute)
            return self.key_attribute
        else:
            assert False