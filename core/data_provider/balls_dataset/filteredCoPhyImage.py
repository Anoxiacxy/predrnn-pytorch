import json
import os
from typing import Dict

import PIL
import imageio
import numpy as np

from .filteredCoPhyVideo import make_2d_grid, add_jitter
from .filteredCoPhyVideo import FilteredCoPhyVideo
from .filteredCoPhyVideo import BallsCFVideoDataset
from .filteredCoPhyVideo import BlocktowerCFVideoDataset
from .filteredCoPhyVideo import CollisionCFVideoDataset


class FilteredCoPhyImage(FilteredCoPhyVideo):
    def get_projection_matrix(self):
        raise NotImplementedError(f"Please Implement this method: {self.get_projection_matrix.__name__}")

    def __init__(self, width=112, height=112, points_subsample=512, jitter=False, **kwargs):
        super(FilteredCoPhyImage, self).__init__(**kwargs)
        self.width = width
        self.height = height
        self.points_subsample = points_subsample
        self.jitter = jitter
        self.points = make_2d_grid([-0.5, -0.5], [0.5, 0.5], [self.height, self.width]).float()
        self.image_index = self.video_index
        self.image_info = json.load(
            open(os.path.join(self.data_location, f'annot_{self.mode}.json')))

    def get_mask_item(self, item) -> Dict:
        info = self.image_info[item]
        mask_path = info['mask_names']
        assert info['image_name'] == f'img_{item:06d}.png'
        # Set additional background slot / mask
        instance_masks = [np.zeros((self.height, self.width)).astype(bool)]

        count = len(mask_path)
        for mask_id in range(count):
            # PIL image mode=1 size=112x112
            m = np.asarray(PIL.Image.open(os.path.join(self.data_location, self.mode, mask_path[mask_id])))
            instance_masks.append(m)

        # (height, width, slots) -> bool
        masks = np.stack(instance_masks, axis=2).astype(np.bool)
        # (height, width) -> slot
        masks = np.argmax(masks, axis=2)

        return {'masks': masks}

    def get_image_item(self, item) -> Dict:
        img_file = f'img_{item:06d}.png'
        img = np.asarray(imageio.imread(os.path.join(self.data_location, self.mode, img_file)))[..., :3]
        img = img.astype(np.float32) / 255
        img_tr = np.transpose(img, (2, 0, 1))

        if self.mode == 'train' and self.jitter:
            assert (self.width == self.height)  # We're being lazy here.
            points = add_jitter(self.points, self.jitter, resolution=self.width)
        else:
            points = self.points

        sample = dict()
        sample.update({
            'inputs': img_tr,  # [3, h, w]
            'points': points,  # [h*w, 2]
            'points.values': np.reshape(img, (self.height * self.width, 3)),  # [h*w, 3]
        })
        return sample


class BallsCFImageDataset(FilteredCoPhyImage, BallsCFVideoDataset):
    def __init__(self, **kwargs):
        super(BallsCFImageDataset, self).__init__(**kwargs)

    def __len__(self):
        return len(self.video_index * 4)

    def __getitem__(self, item):
        data = dict()
        data.update(self.get_image_item(item))
        data.update(self.get_mask_item(item))
        return data

    def get_projection_matrix(self):
        return super().get_projection_matrix()

class BlocktowerCFImageDataset(BlocktowerCFVideoDataset, FilteredCoPhyImage):
    def __init__(self, **kwargs):
        super(BlocktowerCFImageDataset, self).__init__(**kwargs)

    def __len__(self):
        return len(self.video_index * 4)

    def __getitem__(self, item):
        data = dict()
        data.update(self.get_image_item(item))
        data.update(self.get_mask_item(item))
        return data

    def get_projection_matrix(self):
        return super().get_projection_matrix()


class CollisionCFImageDataset(CollisionCFVideoDataset, FilteredCoPhyImage):
    def __init__(self, **kwargs):
        super(CollisionCFImageDataset, self).__init__(**kwargs)

    def __len__(self):
        return len(self.video_index * 4)

    def __getitem__(self, item):
        data = dict()
        data.update(self.get_image_item(item))
        data.update(self.get_mask_item(item))
        return data

    def get_projection_matrix(self):
        return super().get_projection_matrix()