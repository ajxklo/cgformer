import numpy as np
import torch
from PIL import Image

data_config={
    'input_size': (384, 1280),
    'resize': (0., 0.),
    'rot': (0.0, 0.0 ),
    'flip': (0.0, 0.0 ),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

def sample_augmentation(H, W, flip=None, scale=None):
    fH, fW = 384,1280

    resize = float(fW) / float(W)
    resize += np.random.uniform(*data_config['resize'])
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    crop_h = int((1 - np.random.uniform(*data_config['crop_h'])) * newH) - fH
    crop_w = int(np.random.uniform(0, max(0, newW - fW)))
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    flip = data_config['flip'] and np.random.choice([0, 1])
    rotate = np.random.uniform(*data_config['rot'])

    return resize, resize_dims, crop, flip, rotate

def img_transform_core(img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        return img

def img_transform(img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran