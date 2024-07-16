import torch
from torchvision import transforms
from torchvision.transforms import functional as vision_F
import numpy as np
from itertools import permutations
import random
import PIL
from copy import deepcopy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

NUM_DISCREATE = 11

def is_color_transform(transform):  
    return transform in ['brightness', 'contrast', 'saturation', 'hue']


def clip(n, min, max):
    if n > max:
        return max
    elif n < min:
        return min
    return n


def ShearX(img, v):  # [-0.3, 0.3]
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Posterize(img, v):  # [4, 8]
    assert 4 <= v <= 8
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f

def Identity(img, v):
    return img




def Brightness_v(img, v): # [0.1,1.9]
    return vision_F.adjust_brightness(img, v)

def Contrast_v(img, v): # [0.1,1.9]
    return vision_F.adjust_contrast(img, v)

def Saturation_v(img, v): # [0.1,1.9]
    return vision_F.adjust_saturation(img, v)

def Hue_v(img, v): # [-0.45, 0.45]
    return vision_F.adjust_hue(img, v)

def Grayscales_v(img, _): # [-0.45, 0.45]
    return vision_F.rgb_to_grayscale(img, num_output_channels=3)




transformations_list = [
        (ShearX, -0.3, 0.3),  # 0
        (ShearY, -0.3, 0.3),  # 1
        (TranslateX, -0.45, 0.45),  # 2
        (TranslateY, -0.45, 0.45),  # 3
        (Rotate, -30, 30),  # 4
        (AutoContrast, 0, 1),  # 5
        (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        (Solarize, 0, 256),  # 8
        (Posterize, 4, 8),  # 9
        (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        (Cutout, 0, 0.2),  # 14
        (Identity, 0, 1)
    ]


# transformations_list = [
#         (Brightness_v, 0.1, 1.9),
#         (Contrast_v, 0.1, 1.9),
#         (Saturation_v, 0.1, 1.9),
#         (Hue_v, -0.45, 0.45),
#     ]

# transformations_list = [
#         (Brightness_v, 0.6, 1.4),
#         (Contrast_v, 0.6, 1.4),
#         (Saturation_v, 0.6, 1.4),
#         (Hue_v, -0.1, 0.1),
#     ]

transformations_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in transformations_list}


def apply_augment(img, name, level):
    augment_fn, low, high = transformations_dict[name]
    return augment_fn(img.copy(), level * (high - low) + low)


def split_interval(lower: float, upper: float, N: int):

    interval_size = (upper - lower) / (N - 1)
    split_points = [round(lower + i * interval_size, 3) for i in range(N)]

    return split_points   


def get_transforms_list(actions):
    
    actions_1 = [action1 for action1, _ in actions]
    actions_2 = [action2 for _, action2 in actions]

    return (
        actions_1, actions_2
    )


class RandomAugmentation(object):
    def __init__(self, N, pr):
        self.N = N
        self.pr = pr

    def __call__(self, img):
        operations = list(transformations_dict.keys())
        transformations_details = []
        
        for _ in range(self.N):
            
            if random.random() > self.pr:
                continue
            
            name = random.choice(operations)
            level = random.random()
            
            img = apply_augment(img, name, level)
            transformations_details.append( (name[:3], round(level,2)) )
            
        return img, transformations_details



class CustomRandomResizedCrop(transforms.RandomResizedCrop):
    def forward(self, img):
        W, H = img.size[-2:]
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        # i, j, h, w = 16, 16, 32, 32
        return vision_F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias), (max(i,0), max(j, 0), min(i+h, H), min(j+w, W))


class CustomRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def forward(self, img):
        if torch.rand(1) < self.p:
            return vision_F.hflip(img), 1
        return img, 0

class Augmentation(object):
    def __init__(self, policies, dist):
        
        assert len(policies) != 0, 'policies should contain at least one policy'
        assert np.isclose(sum(dist), 1), 'probabilities do not sum to 1'
        assert len(policies) >= len(dist), 'len(policies) must be greater than len(dist)'
        
        self.policies = policies
        self.dist = dist
        
        
    
    def get_policy(self, dist):
        idx = np.random.choice(range(len(dist)), p=dist)
        policy = self.policies[-(idx+1)]
        return policy
    
    def __call__(self, img, branch=None):
        
        policy = self.get_policy(self.dist)
        
        img = img.copy()
        subpolicy_1, subpolicy_2 = random.choice(policy)

        if branch == 1:
            subpolicy_1, _ = random.choice(policy)
            img = img.copy()
            for name, pr, lvl in subpolicy_1:
                if random.random() < pr:
                    img = apply_augment(img, name, lvl)
            return img
        
        elif branch == 2:
            _, subpolicy_2 = random.choice(policy)
            img = img.copy()
            for name, pr, lvl in subpolicy_2:
                if random.random() < pr:
                    img = apply_augment(img, name, lvl)
            return img
        
        elif branch is None:
            subpolicy_1, subpolicy_2 = random.choice(policy)
            img1, img2 = img.copy(), img.copy()
            
            for name, pr, lvl in subpolicy_1:
                if random.random() < pr:
                    img1 = apply_augment(img1, name, lvl)
            
            for name, pr, lvl in subpolicy_2:
                if random.random() < pr:
                    img2 = apply_augment(img2, name, lvl)
            
            return img1, img2

def apply_transformations(img1, transform_list):
        
    num_samples = len(img1)
    stored_imgs = []
    
    for i in range(num_samples):
        img = img1[i]
        for (name, pr, level) in transform_list[i]:
            if random.random() < pr:
                assert 0 <= level <= 1
                img = apply_augment(img, name, level)
        
        
        stored_imgs.append(img)
    
    return stored_imgs

def get_policy_distribution(N, p):
    return [p*(1-p)**n/(1-(1-p)**N) for n in range(N)]
    


