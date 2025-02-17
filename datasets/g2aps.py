# encoding: utf-8
"""
@author:  huynguyen792
@contact: nguyet91@qut.edu.au
"""

import logging

import json

import os.path as osp

from prettytable import PrettyTable


class G2APS(object):
    logger = logging.getLogger("IRRA.dataset")
    dataset_dir = 'G2APS-AG'

    def __init__(self, root='./data',
                 verbose=True, type="dual",text="aerial", **kwargs):
        super(G2APS, self).__init__()
        self.type=type
        self.text = text
        self.flag = 0
        # if name == "G2APS":
        #     camera = [0,1]
        # elif name == "G2APS-g":
        #     camera = [0]
        # else:
        #     camera = [1]
        camera = [1]

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'img')

        self.query_dir = osp.join(self.dataset_dir, 'img')
        self.gallery_dir = osp.join(self.dataset_dir, 'img')

        with open(osp.join(self.dataset_dir,"train.json"),"r") as f:
            data_train = json.load(f)

        with open(osp.join(self.dataset_dir,"test.json"),"r") as f:
            data_test = json.load(f)

        with open(osp.join(self.dataset_dir,"g2aps.json"),"r") as f:
            data_text = json.load(f)

        id_list = [int(i[0]) for i in data_train]+[int(i[0]) for i in data_test]
        self.id_key = {key:i for i,key in enumerate(set(id_list))}

        d = {}
        for item in data_text:
            d.update(item)
        self.text_captions = d

        self._check_before_run()

        self.train, self.train_id_container = self._process_dir(data_train, is_train=True, camera=camera)
        self.test, self.test_id_container = self._process_dir(data_test, is_train=False, camera=camera)
        self.val, self.val_id_container = self._process_dir(data_test, is_train=False, camera=camera)

        if verbose:
            self.logger.info("=> AGTBPR Images and Captions are loaded")
            self.show_dataset_info()

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = \
            len(self.train_id_container), len(self.train), len(self.text_captions)
        num_test_pids, num_test_imgs, num_test_captions = \
            len(self.test_id_container), len(self.test['img_paths']), len(self.test['img_paths'])
        num_val_pids, num_val_imgs, num_val_captions = \
            len(self.val_id_container), len(self.val['img_paths']), len(self.val['img_paths'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        print(table)
        self.logger.info("\n"+str(table))

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, data, is_train=True, camera=[1]):     
        pid_dict = {key:[] for key in range(len(self.id_key))}
        pid_dict_ptr = {key:0 for key in range(len(self.id_key))}
        # Set pid_dict
        for i,item in enumerate(data):
            p_id, _, img_path, isuav = item
            p_id = self.id_key[int(p_id)]
            if isuav not in camera:
                pid_dict[int(p_id)].append(i)

        # 设置输入pair，分别是人id，图像id，图像path，GT图像path，以及是否是uav
        data_list = []
        for i,item in enumerate(data):
            p_id, _, img_path, isuav = item
            p_id = self.id_key[int(p_id)]
            if isuav in camera and len(pid_dict[int(p_id)]) != 0:
                temp_data = [p_id, i, img_path, data[pid_dict[p_id][pid_dict_ptr[p_id]]][2], isuav]
                data_list.append(temp_data)
                pid_dict_ptr[p_id] = (pid_dict_ptr[p_id]+1) % len(pid_dict[p_id])

        pid_container = set()    
        dataset = []
        image_pids = []
        image_paths = []
        caption_pids = []
        captions = []
        gnd_img_paths = []

        # process data
        image_id = 0
        for item in data_list:
            pid,_,img_path,img_path_pair,isuav = item
            pid_container.add(pid)
            if self.text=="aerial":
                key = img_path
            elif self.text=="ground":
                key = img_path_pair
            elif self.text=="half":
                key = img_path if self.flag else img_path_pair
                self.flag = (self.flag+1) % 2
            else:
                raise NotImplementedError("no supportted text type{}".format(self.text))
            text = self.text_captions[key]
            img_path = osp.join(self.train_dir,img_path)
            img_path_pair = osp.join(self.train_dir,img_path_pair)
            if is_train:
                dataset.append((pid, image_id, img_path,img_path_pair, text))
                image_id += 1
            else:
                #img_path = osp.join(self.train_dir,img_path)
                image_pids.append(pid)
                image_paths.append(img_path)
                gnd_img_paths.append(img_path_pair)
                caption_pids.append(pid)
                captions.append(text)
            pass


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


if __name__ == '__main__':
    save_path = r"E:\Share\jupyterDir\UAV-GA-TBPR"
    data = G2APS(root=save_path,name="G2APS-g")
    data.show_dataset_info()