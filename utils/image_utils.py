import torch
import numpy as np
import pickle
import cv2
from skimage.color import rgb2lab
import math

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png", "jpg"])

def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])

def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict    

def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)    

def load_npy(filepath):
    img = np.load(filepath)
    return img

def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

    #    img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    img = img/255.
    return img

def load_resize_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    img = img/255.
    return img

def load_val_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    resized_img = img.astype(np.float32)
    resized_img = resized_img/255.
    return resized_img

def load_mask(filepath, size=None):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # kernel = np.ones((8,8), np.uint8)
    # erosion = cv2.erode(img, kernel, iterations=1)
    # dilation = cv2.dilate(img, kernel, iterations=1)
    # contour = dilation - erosion
    img = img.astype(np.float32)
    # contour = contour.astype(np.float32)
    # contour = contour/255.
    img = img/255.
    return img

def load_resize_mask(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # kernel = np.ones((8,8), np.uint8)
    # erosion = cv2.erode(img, kernel, iterations=1)
    # dilation = cv2.dilate(img, kernel, iterations=1)
    # contour = dilation - erosion
    img = img.astype(np.float32)
    # contour = contour.astype(np.float32)
    # contour = contour/255.
    img = img/255.
    return img

def load_val_mask(filepath):
    img = cv2.imread(filepath, 0)
    resized_img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    resized_img = resized_img.astype(np.float32)
    resized_img = resized_img/255.
    return resized_img

def save_img(img, filepath):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def myPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def batch_PSNR(img1, img2, average=True):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR) if average else sum(PSNR)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):

        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            # image_numpy = image_numpy.convert('L')

        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    # image_numpy =
    return np.clip(image_numpy, 0, 255).astype(imtype)

def calc_RMSE(real_img, fake_img):
    # convert to LAB color space
    real_lab = rgb2lab(real_img)
    fake_lab = rgb2lab(fake_img)
    return real_lab - fake_lab

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

# def yCbCr2rgb(input_im):
#     im_flat = input_im.contiguous().view(-1, 3).float()
#     mat = torch.tensor([[1.164, 1.164, 1.164],
#                        [0, -0.392, 2.017],
#                        [1.596, -0.813, 0]])
#     bias = torch.tensor([-16.0/255.0, -128.0/255.0, -128.0/255.0])
#     temp = (im_flat + bias).mm(mat)
#     out = temp.view(3, list(input_im.size())[1], list(input_im.size())[2])


def splitimage(imgtensor, crop_size=128, overlap_size=64):
    _, C, H, W = imgtensor.shape
    hstarts = [x for x in range(0, H, crop_size - overlap_size)]
    while hstarts and hstarts[-1] + crop_size >= H:
        hstarts.pop()
    hstarts.append(H - crop_size)
    wstarts = [x for x in range(0, W, crop_size - overlap_size)]
    while wstarts and wstarts[-1] + crop_size >= W:
        wstarts.pop()
    wstarts.append(W - crop_size)
    starts = []
    split_data = []
    for hs in hstarts:
        for ws in wstarts:
            cimgdata = imgtensor[:, :, hs:hs + crop_size, ws:ws + crop_size]
            starts.append((hs, ws))
            split_data.append(cimgdata)
    return split_data, starts

def get_scoremap(H, W, C, B=1, is_mean=True):
    center_h = H / 2
    center_w = W / 2

    score = torch.ones((B, C, H, W))
    if not is_mean:
        for h in range(H):
            for w in range(W):
                score[:, :, h, w] = 1.0 / (math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-6))
    return score

def mergeimage(split_data, starts, crop_size = 128, resolution=(1, 3, 128, 128)):
    B, C, H, W = resolution[0], resolution[1], resolution[2], resolution[3]
    tot_score = torch.zeros((B, C, H, W))
    merge_img = torch.zeros((B, C, H, W))
    scoremap = get_scoremap(crop_size, crop_size, C, B=B, is_mean=False)
    for simg, cstart in zip(split_data, starts):
        hs, ws = cstart
        merge_img[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap * simg
        tot_score[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap
    merge_img = merge_img / tot_score
    return merge_img