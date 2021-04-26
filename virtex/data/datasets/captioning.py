import os
import os.path as osp
import random
from typing import Callable, Dict, List

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image


import albumentations as alb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from virtex.data.readers import LmdbReader
from virtex.data.tokenizers import SentencePieceBPETokenizer
from virtex.data import transforms as T


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets
        self.collate_fn = datasets[0].collate_fn

    @property
    def name(self):
        return '_'.join([dset.name for dset in self.datasets])
    
    @property
    def tokenizer(self):
        return self.datasets[0].tokenizer
    
    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])
    
    def __getitem__(self, idx):
        old_idx = idx
        for dataset in self.datasets:
            if idx < len(dataset):
                break
            idx -= len(dataset)
        data = dataset[idx]
        data['image_id'] = torch.tensor(old_idx, dtype=torch.long)
        return data
    

class CaptionDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        tokenizer: SentencePieceBPETokenizer,
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
        max_caption_length: int = 77,
        percentage: float = 100.0,
        all_captions: bool = False,
        include_image: bool = True
    ):
        super().__init__()
        self.percentage = percentage
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.caption_transform = alb.Compose(
            [
                T.NormalizeCaption(),
                T.TokenizeCaption(tokenizer),
                T.TruncateCaptionTokens(max_caption_length),
            ]
        )
        self.padding_idx = tokenizer.pad_id
        self.data_root = data_root
        self.split = split
        
        self._all_captions = all_captions
        self._include_image = include_image

    @property
    def name(self):
        raise NotImplementedError

    @property
    def size(self):
        raise NotImplementedError

    def __len__(self):
        return int(self.size * self.percentage / 100.)

    def _get_data(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        data = self._get_data(idx)
        caption = data['caption']

        if self._include_image:
            image = data['image']
            image_caption = self.image_transform(image=image, caption=caption)
            image, caption = image_caption["image"], image_caption["caption"]
            image = np.transpose(image, (2, 0, 1))

            data['image'] = torch.tensor(image, dtype=torch.float)

        if isinstance(caption, list):
            # only used for eval
            caption_tokens = [self.caption_transform(caption=c)['caption']
                              for c in caption]
            caption = [self.tokenizer.decode(c) for c in caption_tokens]
            data['caption'] = caption
        else:
            caption_tokens = self.caption_transform(caption=caption)["caption"]
            del data['caption']

            data.update({
                "caption_tokens": torch.tensor(caption_tokens, dtype=torch.long),
                "noitpac_tokens": torch.tensor(caption_tokens, dtype=torch.long).flip(0),
                "caption_lengths": torch.tensor(len(caption_tokens), dtype=torch.long)
            })

        return data

    def collate_fn(
        self, data: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:

        batch_data = dict()
        for k in data[0].keys():
            if k in ['caption_tokens', 'noitpac_tokens']:
                batch_datum = torch.nn.utils.rnn.pad_sequence(
                    [d[k] for d in data],
                    batch_first=True,
                    padding_value=self.padding_idx
                )
            elif k == 'caption':
                batch_datum = [d[k] for d in data]
            else:
                batch_datum = torch.stack([d[k] for d in data], dim=0)
            batch_data[k] = batch_datum

        return batch_data
        

class CocoDataset(CaptionDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        root = osp.join(self.data_root, f'{self.split}2017')
        ann_file = osp.join(self.data_root, 'annotations', f'captions_{self.split}2017.json')

        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.root = root

    @property
    def name(self):
        return 'coco'

    @property
    def size(self):
        return len(self.ids)
    
    def _get_data(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        caption = [ann['caption'] for ann in anns]

        if not self._all_captions:
            caption = random.choice(caption)

        data = {'image_id': torch.tensor(idx, dtype=torch.long),
                'caption': caption}

        if self._include_image:
            path = coco.loadImgs(img_id)[0]['file_name']
            image = Image.open(os.path.join(self.root, path)).convert('RGB')
            image = np.array(image)
            data['image'] = image

        return data

        
class ConceptualCaptionsDataset(CaptionDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_file = osp.join(self.data_root, f'{self.split}_data2.tsv')
        self.data = pd.read_csv(data_file, sep='\t')

    @property
    def name(self):
        return 'cc'

    @property
    def size(self):
        return len(self.data)
    
    def _get_data(self, idx):
        row = self.data.iloc[idx]

        caption = row['caption']
        if self._all_captions:
            caption = [caption]

        data = {'image_id': torch.tensor(idx, dtype=torch.long),
                'caption': caption}

        if self._include_image:
            image_filename = row['filename']
            image = Image.open(osp.join(self.data_root, image_filename))
            image = np.array(image.convert("RGB"))
            data['image'] = image

        return data
