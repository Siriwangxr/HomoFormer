import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from utils import is_png_file, load_img, Augment_RGB_torch, load_val_img, load_mask, load_val_mask, load_resize_img, load_resize_mask
import torch.nn.functional as F
import random
import cv2

augment = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None, plus=False):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform
        gt_dir = 'train_C'
        input_dir = 'train_A'
        mask_dir = 'train_B'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        mask_files = sorted(os.listdir(os.path.join(rgb_dir, mask_dir)))
        
        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files]
        self.mask_filenames = [os.path.join(rgb_dir, mask_dir, x) for x in mask_files]

        self.img_options = img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target
        self.img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

        self.homo_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean_0 = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy_0 = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        mask = load_mask(self.mask_filenames[tar_index])
        mask = torch.from_numpy(np.float32(mask))

        clean_0 = clean_0.permute(2,0,1).contiguous()
        noisy_0 = noisy_0.permute(2,0,1).contiguous()


        sam_img_SR = self.img_transform(noisy_0)
        sam_img_mask = self.mask_transform(mask)

        noisy = self.homo_transform(noisy_0)
        clean = self.homo_transform(clean_0)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        mask_filename = os.path.split(self.mask_filenames[tar_index])[-1]

        clean_filename = clean_filename.split(".")[0].replace("_free", "")

        # #Crop Input and Target
        # ps = self.img_options['patch_size']
        # H = clean.shape[1]
        # W = clean.shape[2]
        # if H-ps==0:
        #     r=0
        #     c=0
        # else:
        #     r = np.random.randint(0, H - ps)
        #     c = np.random.randint(0, W - ps)
        # clean = clean[:, r:r + ps, c:c + ps]
        # noisy = noisy[:, r:r + ps, c:c + ps]
        # mask = mask[r:r + ps, c:c + ps]
        # random_number = random.getrandbits(3)
        # apply_trans = transforms_aug[random_number]
        #
        # clean = getattr(augment, apply_trans)(clean)
        # noisy = getattr(augment, apply_trans)(noisy)
        # mask = getattr(augment, apply_trans)(mask)
        mask = torch.unsqueeze(mask, dim=0)
        mask = self.homo_transform(mask)





        return {'HR':clean, 'SR': noisy, 'mask': mask,
                'filename': clean_filename,
                'sam_SR': sam_img_SR, 'sam_mask': sam_img_mask}


##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None, plus=False):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform
        # if plus:
        #     gt_dir = 'test_C_fixed_official'
        # else:
        gt_dir = 'test_C'
        input_dir = 'test_A'
        mask_dir = 'test_B'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        mask_files = sorted(os.listdir(os.path.join(rgb_dir, mask_dir)))

        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_png_file(x)]
        self.mask_filenames = [os.path.join(rgb_dir, mask_dir, x) for x in mask_files if is_png_file(x)]

        self.tar_size = len(self.clean_filenames)

        self.img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

        self.homo_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])


    def __len__(self):
        return self.tar_size
        # return 3

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        mask = load_mask(self.mask_filenames[tar_index])
        mask = torch.from_numpy(np.float32(mask))

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        mask_filename = os.path.split(self.mask_filenames[tar_index])[-1]

        clean_filename = clean_filename.split(".")[0].replace("_free", "")

        clean = clean.permute(2,0,1).contiguous()
        noisy = noisy.permute(2,0,1).contiguous()
        mask = torch.unsqueeze(mask, dim=0)

        clean = self.homo_transform(clean)
        noisy = self.homo_transform(noisy)
        _mask = self.homo_transform(mask)

        sam_img_SR = self.img_transform(noisy)
        sam_img_mask = self.mask_transform(mask)

        # return clean, noisy, mask, clean_filename, noisy_filename
        return {'HR':clean, 'SR': noisy, 'mask': _mask,
                'filename': clean_filename,
                'sam_SR': sam_img_SR, 'sam_mask': sam_img_mask}


##################################################################################################
class DataLoaderSBUVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderSBUVal, self).__init__()

        self.target_transform = target_transform

        # gt_dir = 'test_C_fixed_official'

        input_dir = 'frames'
        mask_dir = 'test_B'

        mask_file = sorted(os.listdir(os.path.join(rgb_dir, mask_dir)))
        clean_files = []
        input_files = []
        mask_files = []
        for file in mask_file:
            for m in os.listdir(os.path.join(rgb_dir, input_dir, file[:-9])):
                input_files.append(os.path.join(rgb_dir, input_dir, file[:-9], m))
                clean_files.append(os.path.join(rgb_dir, input_dir, file[:-9], m))
                mask_files.append(os.path.join(rgb_dir, mask_dir,  file))

        self.clean_filenames = [x for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [x for x in input_files if is_png_file(x)]
        self.mask_filenames = [x for x in mask_files if is_png_file(x)]

        self.tar_size = len(self.clean_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        # clean = torch.from_numpy(np.float32(load_resize_img(self.clean_filenames[tar_index])))
        # noisy = torch.from_numpy(np.float32(load_resize_img(self.noisy_filenames[tar_index])))
        # mask = load_resize_mask(self.mask_filenames[tar_index])

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        mask = load_mask(self.mask_filenames[tar_index], size=(640, 480))
        mask = torch.from_numpy(np.float32(mask))

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        mask_filename = os.path.split(self.mask_filenames[tar_index])[-1]

        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)
        mask = torch.unsqueeze(mask, dim=0)

        return clean, noisy, mask, clean_filename, noisy_filename, mask_filename
