from typing import List
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import regex as re
import copy

tmp_sent = ["The person in the picture is {}.","This person is {}."]
tmp_sent_light = ["The light in the picture is {}.","This picture is taken in {} condition."]
tmp_sent_angle = ["The person in the picture is {}.", "The picture is taken from {}."]

sentence_list = [
    [###gender: male=0, female = 1
        [t_s.format(kw) for t_s in tmp_sent for kw in ["male","a man","not female", "not a woman"]],
        [t_s.format(kw) for t_s in tmp_sent for kw in ["female","a woman","not male", "not a man"]]
    ],
    [### age: young=1, old = 2, adult=0
        [t_s.format(kw) for t_s in tmp_sent for kw in ["adult","adulthood","come of age","middle age","middlescence", "middle-aged"]],
        [t_s.format(kw) for t_s in tmp_sent for kw in ["youth","youngster","the young", "young"]],
        [t_s.format(kw) for t_s in tmp_sent for kw in ["old","wrinkly","the older"]]
    ],
    [### light: # dim=1, bright=0
        [t_s.format(kw) for t_s in tmp_sent_light for kw in ["bright","brilliant","light", "lighting", "brightness"]],
        [t_s.format(kw) for t_s in tmp_sent_light for kw in ["dark","dim","half-light", "gloom"]]
    ],
    [### weight: slim=1, fat=2, middle=0
        [t_s.format(kw) for t_s in tmp_sent for kw in ["medium built", "average weight", "middle weight", "medium size"]],
        [t_s.format(kw) for t_s in tmp_sent for kw in ["thin","slim","slender", "lean"]],
        [t_s.format(kw) for t_s in tmp_sent for kw in ["fat","solid"]]
    ],
    [### height: tall=1, short=2, side=0
        [t_s.format(kw) for t_s in tmp_sent for kw in ["average height", "not tall or short", "medium height"]],
        [t_s.format(kw) for t_s in tmp_sent for kw in ["tall"]],
        [t_s.format(kw) for t_s in tmp_sent for kw in ["short"]]
    ],
    [### angle: # front=1, back = 2, other=0
        [tmp_sent_angle[1].format(kw) for kw in ["the side","side-shot"]],
        [tmp_sent_angle[0].format(kw) for kw in ["facing the camera",]]+\
        [tmp_sent_angle[1].format(kw) for kw in ["the front",]],
        [tmp_sent_angle[0].format(kw) for kw in ["away from the camera",]]+\
        [tmp_sent_angle[1].format(kw) for kw in ["the back",]],
    ],
    [### attitude: # Aerial=1, ground=0
        ["This person seems to be at the same altitude as camera.",
         "This person is at the horizontal position of the camera."],
        ["This photo is from a top view.",
         "The picture was taken from a height down.",
        "This person's horizontal position is below the camera."]
    ],
]

class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("IRRA.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item  = self.dataset[index]
        if len(item)==5:
            pid, image_id, img_path, pair_img_path, caption = item
        elif len(item)==19:
            pid, image_id, img_path, pair_img_path, caption = item[:5]
            atr_a = item[5:12]
            atr_g = item[12:19]

        img = read_image(img_path)
        pair_img = read_image(pair_img_path) if pair_img_path is not None else None
        if self.transform is not None:
            img = self.transform(img)
            pair_img = self.transform(pair_img) if pair_img is not None else None

        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'pair_img': pair_img,
            'caption_ids': tokens,
        }

        if len(item)==19:
            neg_sent_list = []
            pos_sent_list = []
            for atr,label in zip(sentence_list,atr_a):
                size = len(atr)
                neg_label = (label+1)%size
                pos_sent_list.append(random.choice(atr[label]))
                neg_sent_list.append(random.choice(atr[neg_label]))
            
            ret['atr_a_pos'] = torch.stack([tokenize(sent, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate) for sent in pos_sent_list])
            ret['atr_a_neg'] = torch.stack([tokenize(sent, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate) for sent in neg_sent_list])
            
            neg_sent_list = []
            pos_sent_list = []
            for atr,label in zip(sentence_list,atr_g):
                size = len(atr)
                neg_label = (label+1)%size
                pos_sent_list.append(random.choice(atr[label]))
                neg_sent_list.append(random.choice(atr[neg_label]))

            ret['atr_g_pos'] = torch.stack([tokenize(sent, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate) for sent in pos_sent_list])
            ret['atr_g_neg'] = torch.stack([tokenize(sent, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate) for sent in neg_sent_list])
            
            ret['atr_a'] = torch.LongTensor(atr_a)
            ret['atr_g'] = torch.LongTensor(atr_g)
            
        return ret


class ImagePairDataset(Dataset):
    def __init__(self, image_pids, img_paths, pair_img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.pair_img_paths = pair_img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path, pair_img_path = self.image_pids[index], self.img_paths[index], self.pair_img_paths[index]
        img = read_image(img_path)
        pair_img = read_image(pair_img_path) if pair_img_path is not None else None
        if self.transform is not None:
            img = self.transform(img)
            pair_img = self.transform(pair_img) if pair_img is not None else None
        return pid, img, pair_img
    
class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption


class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate

        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item  = self.dataset[index]
        if len(item)==5:
            pid, image_id, img_path, pair_img_path, caption = item
        elif len(item)==19:
            pid, image_id, img_path, pair_img_path, caption = item[:5]
            atr_a = item[5:12]
            atr_g = item[12:19]
        img = read_image(img_path)
        pair_img = read_image(pair_img_path) if pair_img_path is not None else None
        if self.transform is not None:
            img = self.transform(img)
            pair_img = self.transform(pair_img) if pair_img is not None else None
        
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'pair_img': pair_img,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels
        }
        if len(item)==19:
            neg_sent_list = []
            pos_sent_list = []
            for atr,label in zip(sentence_list,atr_a):
                size = len(atr)
                neg_label = (label+1)%size
                pos_sent_list.append(random.choice(atr[label]))
                neg_sent_list.append(random.choice(atr[neg_label]))
            
            ret['atr_a_pos'] = torch.stack([tokenize(sent, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate) for sent in pos_sent_list])
            ret['atr_a_neg'] = torch.stack([tokenize(sent, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate) for sent in neg_sent_list])
            
            neg_sent_list = []
            pos_sent_list = []
            for atr,label in zip(sentence_list,atr_g):
                size = len(atr)
                neg_label = (label+1)%size
                pos_sent_list.append(random.choice(atr[label]))
                neg_sent_list.append(random.choice(atr[neg_label]))

            ret['atr_g_pos'] = torch.stack([tokenize(sent, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate) for sent in pos_sent_list])
            ret['atr_g_neg'] = torch.stack([tokenize(sent, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate) for sent in neg_sent_list])
            
            ret['atr_a'] = torch.LongTensor(atr_a)
            ret['atr_g'] = torch.LongTensor(atr_g)

        return ret

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)