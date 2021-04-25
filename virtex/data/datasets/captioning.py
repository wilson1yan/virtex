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
from torchvision.datasets.coco import CocoCaptions

from virtex.data.readers import LmdbReader
from virtex.data.tokenizers import SentencePieceBPETokenizer
from virtex.data import transforms as T


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets
        self.collate_fn = datasets[0].collate_fn
    
    @property
    def tokenizer(self):
        return self.datasets[0].tokenizer
    
    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])
    
    def __getitem__(self, idx):
        for dataset in self.datasets:
            if idx < len(dataset):
                break
            idx -= len(dataset)
        return dataset[idx]
    

class CaptionDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        tokenizer: SentencePieceBPETokenizer,
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
        max_caption_length: int = 77,
        percentage: float = 100.0,
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

    @property
    def size(self):
        raise NotImplementedError

    def __len__(self):
        return int(self.size * self.percentage / 100.)

    def _get_data(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        data = self._get_data(idx)
        image, caption = data['image'], data['caption']
        image_caption = self.image_transform(image=image, caption=caption)
        image, caption = image_caption["image"], image_caption["caption"]
        image = np.transpose(image, (2, 0, 1))

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
                "image": torch.tensor(image, dtype=torch.float),
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
    def __init__(self, *args, all_captions=False, **kwargs):
        super().__init__(*args, **kwargs)
        root = osp.join(self.data_root, f'{self.split}2017')
        ann_file = osp.join(self.data_root, 'annotations', f'captions_{self.split}2017.json')
        self.coco = CocoCaptions(root, ann_file)
        self._all_captions = all_captions

    @property
    def size(self):
        return len(self.coco)
    
    def _get_data(self, idx):
        image_id = self.coco.ids[idx]
        image, caption = self.coco[idx]

        image = np.array(image)
        if not self._all_captions:
            caption = random.choice(caption)

        return {
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "image": image,
            "caption": caption
        }

        
class ConceptualCaptionsDataset(CaptionDataset):
    def __init__(self, *args, all_captions=False, **kwargs):
        super().__init__(*args, **kwargs)
        data_file = osp.join(self.data_root, f'{self.split}_data2.tsv')
        self.data = pd.read_csv(data_file, sep='\t')
        self._all_captions = all_captions

    @property
    def size(self):
        return len(self.data)
    
    def _get_data(self, idx):
        row = self.data.iloc[idx]

        image_filename = row['filename']
        image = Image.open(osp.join(self.data_root, image_filename))
        image = np.array(image.convert("RGB"))

        caption = row['caption']
        if self._all_captions:
            caption = [caption]

        return {
            "image_id": torch.tensor(0, dtype=torch.long),
            "image": image,
            "caption": caption
        }
