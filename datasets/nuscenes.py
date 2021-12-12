import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from utils import read_list_from_file
from transforms import ResizeWithIntrinsic, RandomHorizontalFlipWithIntrinsic, EqualizeHist, CenterCropWithIntrinsic, map_pt_with_intrinsics
from torchvision.transforms import ToTensor
from .common import NUSCENES_ROOT

import pickle

# full size
_FULL_SIZE = (384, 192)#(768, 384)
# half size
_HALF_SIZE = (384, 192)
# limit of equ
_EQU_LIMIT = 0.004
# robotcar size
_ROBOTCAR_SIZE = _HALF_SIZE#_FULL_SIZE#(576, 320)


#
# Data set
#
class nuScenesSequence(Dataset):
    """
    Oxford RobotCar data set.
    """
    def __init__(self, weather, frame_ids: (list, tuple), augment=True, down_scale=False, num_out_scales=5,
                 gen_equ=False, equ_limit=_EQU_LIMIT, resize=True):
        """
        Initialize
        :param weather: day or night
        :param frame_ids: index of frames
        :param augment: whether to augment
        :param down_scale: whether to down scale images to half of that before
        :param num_out_scales: number of output scales
        :param gen_equ: whether to generate equ image
        :param equ_limit: limit of equ
        :param resize: whether to resize to the same size as robotcar
        """
        resize = True
        # set parameters
        self._root_dir = NUSCENES_ROOT['sequence']
        self._frame_ids = frame_ids
        self._need_augment = augment
        self._num_out_scales = num_out_scales
        self._gen_equ = gen_equ
        self._equ_limit = equ_limit
        self._need_resize = resize
        self._down_scale = down_scale and (not resize)
        if self._down_scale:
            self._width, self._height = _HALF_SIZE
        else:
            self._width, self._height = _FULL_SIZE
        # read all chunks
        if weather in ['day', 'night']:
            chunks = self.read_chunks(os.path.join(NUSCENES_ROOT['split'], '{}_train_split.txt'.format(weather)))
        elif weather == 'both':
            day_chunks = self.read_chunks(os.path.join(NUSCENES_ROOT['split'], 'day_train_split.txt'))
            night_chunks = self.read_chunks(os.path.join(NUSCENES_ROOT['split'], 'night_train_split.txt'))
            chunks = day_chunks + night_chunks
        else:
            raise ValueError(f'Unknown weather parameter: {weather}.')
        # get sequence
        self._sequence_items = self.make_sequence(chunks)
        # transforms
        self._to_tensor = ToTensor()
        if self._need_augment:
            #raise NotImplementedError("Augmentation of point correspondence is not implemented")
            self._flip = RandomHorizontalFlipWithIntrinsic(0.5)
        # crop
        if self._need_resize:
            self._crop = None
            print("[WARINING] Manually disabled cropping")
            #self._crop = CenterCropWithIntrinsic(round(1.8 * self._height), self._height)
        else:
            self._crop = None
        # resize
        if self._down_scale:
            self._resize = ResizeWithIntrinsic(*_HALF_SIZE)
        elif self._need_resize:
            self._resize = ResizeWithIntrinsic(*_ROBOTCAR_SIZE)
        else:
            self._resize = None
        # print message
        print('Frames: {}, Augment: {}, DownScale: {}, '
              'Equ_Limit: {}.'.format(frame_ids, augment, self._down_scale, self._equ_limit))
        print('Total items: {}.'.format(len(self)))

    def read_chunks(self, split_file):
        result = []
        scenes = read_list_from_file(split_file, 1)
        for scene in scenes:
            scene_path = os.path.join(self._root_dir, scene)
            colors = sorted(read_list_from_file(os.path.join(scene_path, 'file_list.txt'), 1))
            colors = [os.path.join(scene, color) for color in colors]
            # read in the point correspondence here
            # dimension is (TxNx5), where T is the frame, N is the number of points
            point_correspondences = []
            for c in colors:
                fn = os.path.basename(c)[:-4] + ".p"
                fn = os.path.join(NUSCENES_ROOT["point_correspondence"], fn)
                with open(fn, 'rb') as f:
                    point_correspondence = pickle.load(f)
                    point_correspondences.append(point_correspondence)
            chunk = {
                'colors': colors,
                'k': np.load(os.path.join(scene_path, 'intrinsic.npy')),
                'ps': point_correspondences
            }
            result.append(chunk)
        return result

    def pack_data(self, src_colors: dict, src_K: np.ndarray, num_scales: int, src_pts):
        out = {}
        h, w, _ = src_colors[0].shape
        # Note: the numpy ndarray and tensor share the same memory!!!
        src_K = torch.from_numpy(src_K)
        # transform
        equ_hist = EqualizeHist(src_colors[0], limit=self._equ_limit)
        # process
        # scale is used in unet, essentially in each "layer"
        # we divide the resolution by half
        for s in range(num_scales):
            # get size
            rh, rw = h // (2 ** s), w // (2 ** s)
            # K and inv_K
            K = src_K.clone()
            if s != 0:
                K[0, :] = K[0, :] * rw / w
                K[1, :] = K[1, :] * rh / h
            out['K', s] = K
            out['inv_K', s] = torch.inverse(K)
            # color
            for fi in self._frame_ids:
                # get color and pts
                color = src_colors[fi]
                equ_color = equ_hist(color)
                pt = src_pts[fi]
                #print(pt.shape)
                #exit()
                # to tensor
                color = self._to_tensor(color)
                equ_color = self._to_tensor(equ_color)
                pt = self._to_tensor(pt)
                pt = pt.squeeze(0)
                pt_prev = pt[:, [1, 2]]
                pt_new = pt[:, [3, 4]]
                # resize
                if s != 0:
                    color = F.interpolate(color.unsqueeze(0), (rh, rw), mode='area').squeeze(0)
                    equ_color = F.interpolate(equ_color.unsqueeze(0), (rh, rw), mode='area').squeeze(0)
                    # manually rescale the point correspondence
                    pt_prev[:, 0] = torch.round(pt_prev[:, 0] * rw / w)
                    pt_prev[:, 1] = torch.round(pt_prev[:, 1] * rh / h)
                    pt_new[:, 0] = torch.round(pt_new[:, 0] * rw / w)
                    pt_new[:, 1] = torch.round(pt_new[:, 1] * rh / h)
                    pt[:, [1, 2]] = pt_prev
                    pt[:, [3, 4]] = pt_new
                # (name, frame_idx, scale)
                out['color', fi, s] = color
                out['color_aug', fi, s] = color
                out["point_correspondence", fi, s] = pt
                #out["pts", s] = pts
                if self._gen_equ:
                    out['color_equ', fi, s] = equ_color
        return out

    def make_sequence(self, chunks: (list, tuple)):
        """
        Make sequence from given folders
        :param chunks:
        :return:
        """
        # store items
        result = []
        # scan
        for chunk in chunks:
            fs = chunk['colors']
            ps = chunk['ps']
            # get length
            frame_length = len(self._frame_ids)
            min_id, max_id = min(self._frame_ids), max(self._frame_ids)
            total_length = len(fs)
            if total_length < frame_length:
                continue
            # pick sequence
            for i in range(abs(min_id), total_length - abs(max_id)):
                items = [fs[i + fi] for fi in self._frame_ids]
                # still an (T_2xNx5) array at this point, T_2 is the number of frames
                point_correspondence = [ps[i + fi] for fi in self._frame_ids]
                point_correspondence = np.asarray(point_correspondence)
                result.append({'sequence': items, 'k': chunk['k'], "pts": point_correspondence})
        return result

    def __getitem__(self, idx):
        """
        Return item according to given index
        :param idx: index
        :return:
        """
        # get item
        item = self._sequence_items[idx]
        # read data
        rgbs = [cv2.imread(os.path.join(self._root_dir, p)) for p in item['sequence']]
        intrinsic = item['k'].copy()
        pts = item['pts']
        pts_result = [np.copy(x) for x in pts]
        # crop
        if self._crop is not None:
            raise NotImplementedError()
            intrinsic, rgbs = self._crop(intrinsic, *rgbs, inplace=False, unpack=False)
        # down scale
        if self._resize is not None:
            intrinsic, rgbs = self._resize(intrinsic, *rgbs)
            # no need to map pts here, as intrinsics is not applied at this stage
            #pts = map_pt_with_intrinsics(pts, intrinsic)
        # augment
        if self._need_augment:
            #raise NotImplementedError()
            intrinsic, rgbs, pts_result = self._flip(intrinsic, *rgbs, unpack=False, pts = pts_result)
        # get colors and pts
        colors = {}
        pts_results = {}
        # color
        for i, fi in enumerate(self._frame_ids):
            colors[fi] = rgbs[i]
            pts_results[fi] = pts_result[i]
        # pack
        result = self.pack_data(colors, intrinsic, self._num_out_scales, pts_results)
        return result

    def __len__(self):
        return len(self._sequence_items)
