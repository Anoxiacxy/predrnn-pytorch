import json
from typing import Dict, Callable, Optional, Tuple

import PIL.Image
import imageio
import numpy as np
import cv2
from numpy import ndarray
from torch.utils.data import DataLoader, Dataset
import os
from random import randint
import torchvision
import pybullet as pb
import torch
from tqdm import tqdm


def load_mask_helper(info: Dict, data_location: str, mode: str, height: int, width: int) -> ndarray:
    mask_paths = info['mask_names']
    # Set additional background slot / mask
    instance_masks = [np.zeros((height, width)).astype(bool)]

    count = len(mask_paths)
    for mask_id in range(count):
        # PIL image mode=1 size=112x112
        m = np.asarray(PIL.Image.open(os.path.join(data_location, mode, mask_paths[mask_id])))
        instance_masks.append(m)

    # (height, width, slots) -> bool
    masks = np.stack(instance_masks, axis=2).astype(np.bool)
    # (height, width) -> slot: int
    masks = np.argmax(masks, axis=2)
    return masks


def load_state_helper(info: Optional, data_location: str, view_matrix, projection_matrix, ex) -> (ndarray, ndarray):
    def _load(pos) -> ndarray:
        states = np.load(os.path.join(data_location, str(ex), pos, 'states.npy'))
        positions = states[..., :3]
        pose_2d = []
        for t in range(positions.shape[0]):
            pose_2d.append([])
            for k in range(positions.shape[1]):
                if not np.all(positions[t, k] == 0):
                    pose_2d[-1].append(convert_to_2d(positions[t, k], view_matrix, projection_matrix, 112))
                else:
                    pose_2d[-1].append(np.zeros(2))
        return np.array(pose_2d)
    return _load('ab'), _load('cd')


class FilteredCoPhyVideo(object):
    def __init__(self, mode="train", resolution=112, load_cd=True, sampling_mode="full",
                 load_ab=True, load_state=False, path="", image_path=None, width=112, height=112, jitter=False,
                 request_raw_data=False, **k):
        """
        :param mode: 'train', 'test' or 'val' split
        :param resolution: Image resolution, default is 112x112
        :param load_ab: if False, the dataloader does not read AB video
        :param load_cd: if False, the dataloader does not read CD video
        :param sampling_mode: 'rand' for random selection in the video, 'fix' for sampling at fixed timestamps and
        'full' for loading the entire video. This is usefull for training De-rendering, where only two images are
        needed instead of the entire video
        :param load_state: True to load the 2D projection of the ground truth state
        """
        super(FilteredCoPhyVideo, self).__init__()
        assert sampling_mode in ['rand', 'fix', 'full', 'fix_6', 'fix_15']

        self.video_index = None
        self.data_location = path
        self.image_path = image_path
        if self.image_path is not None:
            self.image_info = json.load(
                open(os.path.join(self.image_path, f'annot_{mode}.json')))
        else:
            print("FilteredCoPhy_video: No image / mask loaded!")

        assert os.path.isdir(self.data_location)

        self.mode = mode
        self.resolution = resolution
        self.load_cd = load_cd
        self.sampling_mode = sampling_mode
        self.load_ab = load_ab
        self.load_state = load_state
        self.video_length = 0
        self.request_raw_data = request_raw_data

        self.height = height
        self.width = width
        self.jitter = jitter
        self.points = make_2d_grid([-0.5, -0.5], [0.5, 0.5], [self.height, self.width]).float()

    def load_index(self, splits_filename):
        with open(f"{splits_filename}_{self.mode}", "r") as file:
            self.video_index = [int(k) for k in file.readlines()]

    def get_projection_matrix(self):
        raise NotImplementedError(f"Please Implement this method: {self.get_projection_matrix.__name__}")

    def get_mask_item(self, item):
        if self.image_path is None:
            return {}

        ex = self.video_index[item]
        ab_begin_ex = item * 4
        ab_end_ex = item * 4 + 1
        cd_middle_ex = item * 4 + 2
        cd_end_ex = item * 4 + 3

        # print(ex, ab_begin_ex, len(self.image_info))

        return {'masks': load_mask_helper(
            info=self.image_info[ab_begin_ex],
            data_location=self.image_path,
            mode=self.mode,
            height=self.height,
            width=self.width)}

    def get_video_item(self, item):
        ex = self.video_index[item]
        out = {'ex': ex}
        pose_2d_ab = None
        pose_2d_cd = None
        r_ab = None
        r_cd = None

        if self.load_state:
            view_matrix, projection_matrix = self.get_projection_matrix()
            pose_2d_ab, pose_2d_cd = load_state_helper(None, self.data_location, view_matrix, projection_matrix, ex)

        if self.load_ab:
            ab = os.path.join(self.data_location, str(ex), "ab", 'rgb.mp4')
            rgb_ab, r_ab = get_rgb(ab, self.sampling_mode, self.video_length)
            out['rgb_ab'] = rgb_ab
            if self.load_state:
                out["pose_2D_ab"] = pose_2d_ab[r_ab, :, :]

        if self.load_cd:
            cd = os.path.join(self.data_location, str(ex), "cd", 'rgb.mp4')
            rgb_cd, r_cd = get_rgb(cd, self.sampling_mode, self.video_length)
            out['rgb_cd'] = rgb_cd
            if self.load_state:
                out["pose_2D_cd"] = pose_2d_cd[r_cd, :, :]

        if self.request_raw_data:
            return out

        # by lyz
        if self.mode == 'train' and self.jitter:
            assert (self.width == self.height)  # We're being lazy here.
            points = add_jitter(self.points, self.jitter, resolution=self.width)
        else:
            points = self.points
        sample = dict()

        sample.update({
            'inputs': out['rgb_ab'].permute(0, 3, 1, 2),  # [t, 3, h, w]
            'frames': r_ab,  # [t]
            'points': points.repeat(out['rgb_ab'].shape[0], 1, 1),  # [t, h*w, 2]
            'points.values': np.reshape(out['rgb_ab'], (out['rgb_ab'].shape[0], self.height * self.width, 3)),
            'ex': out['ex'],
        })

        return sample


class BallsCFVideoDataset(FilteredCoPhyVideo, Dataset):
    def __init__(self, **kwargs):
        super(BallsCFVideoDataset, self).__init__(**kwargs)
        self.load_index(os.path.join(os.path.dirname(__file__), "filteredCoPhy", "ballsCF_4"))
        self.video_length = 150

    def __len__(self):
        return len(self.video_index)

    def __getitem__(self, item):
        data = dict()
        data.update(self.get_video_item(item))
        data.update(self.get_mask_item(item))
        return data

    def get_projection_matrix(self):
        view_matrix = np.array(pb.computeViewMatrix([0, 0.01, 8], [0, 0, 0], [0, 0, 1])).reshape(
            (4, 4)).transpose()
        projection_matrix = np.array(pb.computeProjectionMatrixFOV(60, 1, 4, 20)).reshape((4, 4)).transpose()
        return view_matrix, projection_matrix


class BlocktowerCFVideoDataset(FilteredCoPhyVideo, Dataset):
    def __init__(self, **kwargs):
        super(BlocktowerCFVideoDataset, self).__init__(**kwargs)
        self.load_index(os.path.join(os.path.dirname(__file__), "filteredCoPhy", "blocktowerCF_4"))
        self.video_length = 150

    def __len__(self):
        return len(self.video_index)

    def __getitem__(self, item):
        data = dict()
        data.update(self.get_video_item(item))
        data.update(self.get_mask_item(item))
        return data

    def get_projection_matrix(self):
        view_matrix = np.array(pb.computeViewMatrix([0, -7, 4.5], [0, 0, 1.5], [0, 0, 1])).reshape(
            (4, 4)).transpose()
        projection_matrix = np.array(pb.computeProjectionMatrixFOV(60, 112 / 112, 4, 20)).reshape((4, 4)).transpose()
        return view_matrix, projection_matrix


class CollisionCFVideoDataset(FilteredCoPhyVideo, Dataset):
    def __init__(self, **kwargs):
        super(CollisionCFVideoDataset, self).__init__(**kwargs)
        self.load_index(os.path.join(os.path.dirname(__file__), "filteredCoPhy", "collisionCF"))
        self.video_length = 75

    def __len__(self):
        return len(self.video_index)

    def __getitem__(self, item):
        data = dict()
        data.update(self.get_video_item(item))
        data.update(self.get_mask_item(item))
        return data

    def get_projection_matrix(self):
        view_matrix = np.array(pb.computeViewMatrix([0, -7, 4.5], [0, 0, 1.5], [0, 0, 1])).reshape(
            (4, 4)).transpose()
        projection_matrix = np.array(pb.computeProjectionMatrixFOV(60, 112 / 112, 4, 20)).reshape((4, 4)).transpose()
        return view_matrix, projection_matrix


def convert_to_2d(pose, view, projection, resolution):
    center_pose = np.concatenate([pose[:3], np.ones((1))], axis=-1).reshape((4, 1))
    center_pose = view @ center_pose
    center_pose = projection @ center_pose
    center_pose = center_pose[:3] / center_pose[-1]
    center_pose = (center_pose + 1) / 2 * resolution
    center_pose[1] = resolution - center_pose[1]
    return center_pose[:2].astype(int).flatten()


def get_rgb(filedir, sampling_mode, video_length):
    assert os.path.exists(filedir)

    if sampling_mode == "full":
        rgb, _, _ = torchvision.io.read_video(filedir, pts_unit="sec")
        rgb = rgb / 255
        # rgb = rgb.permute(0, 3, 1, 2)
        r = list(range(video_length))  # by lyz
    elif sampling_mode == "fix" or sampling_mode == "fix_6":
        rgb, _, _ = torchvision.io.read_video(filedir, pts_unit="sec")
        rgb = rgb[0::int(np.ceil(video_length / 6))]  # each video 6 frames
        rgb = rgb / 255
        # rgb = rgb.permute(0, 3, 1, 2)
        r = list(range(0, video_length, int(np.ceil(video_length / 6))))  # by lyz
    elif sampling_mode == "fix_15":
        rgb, _, _ = torchvision.io.read_video(filedir, pts_unit="sec")
        rgb = rgb[0::int(np.ceil(video_length / 15))]  # each video 15 frames
        rgb = rgb / 255
        # rgb = rgb.permute(0, 3, 1, 2)
        r = list(range(0, video_length, int(np.ceil(video_length / 15))))  # by lyz
    else:
        t = randint(0, int(0.15 * video_length)) if sampling_mode == "rand" else int(0.15 * video_length)
        r = [t, t + int(0.15 * video_length)]
        capture = cv2.VideoCapture(filedir)
        list_rgb = []
        for i in r:
            capture.set(1, i)
            ret, frame = capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            list_rgb.append(frame)
        rgb = np.stack(list_rgb, 0)
        rgb = rgb / 255  # by lyz rgb = 2 * (rgb / 255) - 1
        rgb = rgb.astype(np.float32)  # .transpose(0, 3, 1, 2)
        rgb = torch.FloatTensor(rgb)
    return rgb, r


def make_2d_grid(bb_min, bb_max, shape):
    size = shape[0] * shape[1]
    rows = np.linspace(bb_min[0], bb_max[0], shape[0])
    cols = np.linspace(bb_min[1], bb_max[1], shape[1])
    rows, cols = np.meshgrid(rows, cols, indexing='ij')
    points = np.stack((rows, cols), -1)
    return torch.tensor(points).view(size, 2)


def add_jitter(points, jitter, resolution=64):
    points = points + torch.normal(torch.zeros_like(points),
                                   torch.ones_like(points) / (resolution * jitter))
    return points


def generate_image_dataset(dataset_class: Callable, mode, data_dir, out_dir, **kwargs):
    dataloader = DataLoader(dataset_class(path=data_dir, mode=mode, **kwargs),
                            batch_size=1, shuffle=False, pin_memory=True, num_workers=1)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    img_count = 0
    for i, x in tqdm(enumerate(dataloader)):
        cv2.imwrite(f'{out_dir}/img_{img_count:06d}.png', x['rgb_ab'][0][0].detach().cpu().numpy())  # bz = 1
        img_count = img_count + 1
        cv2.imwrite(f'{out_dir}/img_{img_count:06d}.png', x['rgb_ab'][0][-1].detach().cpu().numpy())  # bz = 1
        img_count = img_count + 1
        B, T, C, H, W = x['rgb_cd'].shape
        cv2.imwrite(f'{out_dir}/img_{img_count:06d}.png', x['rgb_cd'][0][int(T / 2)].detach().cpu().numpy())  # bz = 1
        img_count = img_count + 1
        cv2.imwrite(f'{out_dir}/img_{img_count:06d}.png', x['rgb_cd'][0][-1].detach().cpu().numpy())  # bz = 1
        img_count = img_count + 1

    print(f'finish {dataset_class.__name__} {mode} set!')


if __name__ == '__main__':
    default_config = {
        'resolution': 112,
        'sampling_mode': 'full',
        'load_ab': True, 'load_cd': True, 'load_state': True,
        'request_raw_data': True
    }

    # generate
    generate_image_dataset(
        dataset_class=CollisionCFVideoDataset,
        mode='val',
        data_dir='../../data/CoPhy_112/collisionCF',
        out_dir='../../data/CoPhy_112_imgs_2/collisionCF/val',
        **default_config
    )

    generate_image_dataset(
        dataset_class=BlocktowerCFVideoDataset,
        mode='val',
        data_dir='../../data/CoPhy_112/blocktowerCF/4/',
        out_dir='../../data/CoPhy_112_imgs_2/blocktowerCF/val',
        **default_config
    )

    generate_image_dataset(
        dataset_class=BallsCFVideoDataset,
        mode='val',
        data_dir='../../data/CoPhy_112/ballsCF/4/',
        out_dir='../../data/CoPhy_112_imgs/ballsCF/val',
        **default_config
    )

    generate_image_dataset(
        dataset_class=CollisionCFVideoDataset,
        mode='train',
        data_dir='../../data/CoPhy_112/collisionCF',
        out_dir='../../data/CoPhy_112_imgs/collisionCF/val',
        **default_config
    )

    generate_image_dataset(
        dataset_class=BlocktowerCFVideoDataset,
        mode='train',
        data_dir='../../data/CoPhy_112/blocktowerCF/4/',
        out_dir='../../data/CoPhy_112_imgs/blocktowerCF/val',
        **default_config
    )

    generate_image_dataset(
        dataset_class=BallsCFVideoDataset,
        mode='train',
        data_dir='../../data/CoPhy_112/ballsCF/4/',
        out_dir='../../data/CoPhy_112_imgs/ballsCF/val',
        **default_config
    )

    # train_dataloader = DataLoader(
    #     collisionCF_Video(mode='train', resolution=112, sampling_mode='full', load_ab=True, load_cd=True,
    #                       load_state=True, path=data_path),
    #     batch_size=1, shuffle=False, pin_memory=True, num_workers=1)
    # for i, x in tqdm(enumerate(train_dataloader)):
    #     B, T, C, H, W = x['rgb_ab'].shape
    #     rgb_ab = x['rgb_ab'].view(-1, 3, 112, 112)
    #     rgb_cd = x['rgb_cd'].view(-1, 3, 112, 112)
    #     PSNR = peak_signal_noise_ratio(rgb_ab[0].detach().cpu().numpy(), rgb_ab[1].detach().cpu().numpy() + 1e-4,
    #                                    data_range=2)
    #     print(PSNR)
    #     writer = cv2.VideoWriter(data_path + 'rgb.mp4',
    #                              cv2.VideoWriter_fourcc(*'mp4v'),
    #                              25,
    #                              (112, 112)
    #                              )
    #     rgb = np.uint8(rgb_ab.detach().cpu().numpy())
    #     rgb = np.swapaxes(rgb, 1, 2)
    #     rgb = np.swapaxes(rgb, 2, 3)
    #     for i in range(25):
    #         writer.write(cv2.cvtColor(rgb[i], cv2.COLOR_BGR2RGB))
    #     writer.release()
    #
    #     states = x['pose_2D_cd']
    #     print(states)
    #     break
