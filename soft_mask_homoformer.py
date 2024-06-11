import os
import cv2
import numpy as np
import torch
from PIL import Image
import hydra
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import utils
from utils.image_utils import splitimage, mergeimage
from utils import Augment_RGB_torch
from losses import CharbonnierLoss
from sam_adapter.sam_adapt import SAM
from sam_adapter.iou_loss import IOU
from homoformer import HomoFormer
from warmup_scheduler import GradualWarmupScheduler
import utils.metrics as Metrics
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from utils.loader import get_training_data, get_validation_data
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from omegaconf import DictConfig


from lightning.pytorch.loggers import TensorBoardLogger


def apply_low_pass_filter(mask: object, kernel_size: object = 3) -> object:
    # Create a Gaussian kernel
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)

    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.to(mask.device)

    # Apply the low-pass filter using convolution
    filtered_mask = F.conv2d(mask, kernel, padding=kernel_size // 2)

    return filtered_mask
def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()


class SamShadowDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage="fit"):
        if stage == "fit":
            self.train_data = get_training_data(self.args.train_dir)
            # self.val_data = get_training_data(self.args.val_dir, img_options_train)
        elif stage == "test":
            self.test_data = get_validation_data(self.args.test_dir)

    def train_dataloader(self):
        dataloader = DataLoader(dataset=self.train_data,
                                batch_size=self.args.batch_size,
                                shuffle=True,
                                num_workers=self.args.train_workers,
                                pin_memory=True,
                                drop_last=False)
        return dataloader

    # def val_dataloader(self):
    #     dataloader = DataLoader(dataset=self.val_data,
    #                             batch_size=1,
    #                             shuffle=False,
    #                             num_workers=self.args.eval_workers,
    #                             pin_memory=True,
    #                             drop_last=False)
    #     return dataloader

    def predict_dataloader(self):
        dataloader = DataLoader(dataset=self.test_data, batch_size=self.args.val_batch_size, shuffle=False, num_workers=8, drop_last=False)
        return dataloader


# define a LightningModule, in which I have Homoformer and SAM_adapter two models
def laplacian_magnitude_loss(soft_mask):
    laplacian_kernel = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=soft_mask.dtype, device=soft_mask.device)
    laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
    laplacian = F.conv2d(soft_mask, laplacian_kernel, padding=1)
    magnitude = torch.abs(laplacian)
    mean_magnitude = torch.mean(magnitude)
    return mean_magnitude


def gradient_orientation_map(input_image):
    sobelx = F.conv2d(input_image, torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                                                dtype=input_image.dtype,
                                                device=input_image.device), padding=1)
    sobely = F.conv2d(input_image, torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]],
                                                dtype=input_image.dtype,
                                                device=input_image.device), padding=1)
    gradient_orientation = torch.atan2(sobely, sobelx)
    return gradient_orientation


def gradient_orientation_loss(soft_mask, shadow_input):
    penumbra_area = soft_mask.detach()
    # set a threshold
    threshold_low = 0.01
    threshold_high = 0.995
    penumbra_area = torch.where((penumbra_area >= threshold_low) & (penumbra_area <= threshold_high), 1.0, 0.0)
    shadow_input = torch.mean(shadow_input, dim=1, keepdim=True)
    shadow_input = F.interpolate(shadow_input, (penumbra_area.shape[2], penumbra_area.shape[3]), mode='bilinear', align_corners=False)
    shadow_input = apply_low_pass_filter(shadow_input)

    gradient_orientation_shadow_input = gradient_orientation_map(shadow_input*penumbra_area)
    gradient_orientation_soft_mask = gradient_orientation_map(soft_mask * penumbra_area)
    # using "+" because the two maps has different orientation
    cosine_similarity = torch.cos(gradient_orientation_shadow_input + gradient_orientation_soft_mask)
    gradient_loss = 1 - cosine_similarity
    gradient_loss = torch.mean(gradient_loss)
    return gradient_loss


def soft_mask_loss_metric(soft_mask, shadow_input):
    laplacian_loss = laplacian_magnitude_loss(soft_mask)
    gradient_loss = gradient_orientation_loss(soft_mask, shadow_input)
    # soft_hard_mask_consistency = soft_hard_mask_consistency(soft_mask, hard_mask)
    return laplacian_loss, gradient_loss


def crop_transform_sam_mask(soft_mask, image_size, crop_position, trans_index, patch_size):
    augment = Augment_RGB_torch()
    transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if
                      not method.startswith('_')]
    ps = patch_size
    batch_size = soft_mask.shape[0]
    cropped_masks = []

    for i in range(batch_size):
        H = image_size[0][i]
        W = image_size[1][i]
        r = crop_position[0][i]
        c = crop_position[1][i]
        mask = F.interpolate(soft_mask[i, :, :, :].unsqueeze(0), (H, W), mode='bilinear')
        mask = mask.squeeze(0)
        mask = mask[:, r:r + ps, c:c + ps]
        apply_trans = transforms_aug[trans_index[i]]
        mask = getattr(augment, apply_trans)(mask)
        cropped_masks.append(mask)

    cropped_masks = torch.stack(cropped_masks, dim=0)
    return cropped_masks


class SamShadow(L.LightningModule):
    def __init__(self, sam_adapter, HomoFormer, args, path=None, steps_per_epoch=168):
        super(SamShadow, self).__init__()
        self.args = args
        self.save_model_name = args.name
        self.automatic_optimization = False
        self.steps_per_epoch = int(steps_per_epoch)
        self.homoformer = HomoFormer(img_size=args.train_ps, embed_dim=args.embed_dim, win_size=args.win_size,
                                    token_projection=args.token_projection)
        self.sam = sam_adapter(args.sam.input_size, args.sam.loss)
        self.criterionBCE = torch.nn.BCEWithLogitsLoss()
        self.criterionIOU = IOU()
        self.criterionHomo = CharbonnierLoss()
        self.homoformer_loss = []
        self.soft_mask_loss = []
        # self.soft_mask_loss = []
        if path is not None:
            self.load_pretrained_models(path)
            # transfer output_hypernetworks_mlps parameter to soft_mask_decoder_adapter
            for i in range(self.sam.mask_decoder.num_mask_tokens):
                self.sam.mask_decoder.soft_mask_decoder_adapter[i].load_state_dict(
                    self.sam.mask_decoder.output_hypernetworks_mlps[i].state_dict()
                )
        self.save_hyperparameters()
        self.freeze_parameters(self.sam.parameters())

        if args.unfreeze_sam_head:
            self.unfreeze_parameters(self.sam.mask_decoder.soft_mask_decoder_adapter.parameters())

    @staticmethod
    def freeze_parameters(params):
        for param in params:
            param.requires_grad = False

    @staticmethod
    def unfreeze_parameters(params):
        for param in params:
            param.requires_grad = True

    def forward(self, train_data):
        sam_input = train_data['sam_SR']
        sam_pred_mask_, sam_pred_soft_mask = self.sam(sam_input)
        sam_pred_soft_mask = torch.sigmoid(sam_pred_soft_mask)
        if not self.args.detach_sam and self.args.unfreeze_sam_head:
            soft_mask = sam_pred_soft_mask
        else:
            soft_mask = sam_pred_soft_mask.detach()
        mask = soft_mask # 256, 256
        target = train_data['HR']
        input_ = train_data['SR']
        # save_image(input_[0], 'input.png')

        if self.current_epoch > 5:
            target, input_, mask = utils.MixUp_AUG().aug(target, input_, mask)
        restored = self.homoformer(input_, mask)
        restored = torch.clamp(restored, 0, 1)
        self.homoformer_loss = self.criterionHomo(restored, target)

        residual = (train_data['HR'] + 1) / 2 - (train_data['SR'] + 1) / 2
        residual = torch.mean(residual, dim=1, keepdim=True)
        residual_mask = torch.where(residual < 0.05, torch.zeros_like(residual), torch.ones_like(residual))
        self.segmentation_loss = self.criterionBCE(sam_pred_soft_mask, residual_mask)
        self.segmentation_loss += _iou_loss(sam_pred_soft_mask, residual_mask)
        laplacian_loss, gradient_ori_loss = soft_mask_loss_metric(sam_pred_soft_mask, train_data['HR'])
        self.soft_mask_loss = 10 * (5 * laplacian_loss + gradient_ori_loss)
        self.sam_backward_loss = self.segmentation_loss + self.args.homoformer_loss_scale * self.homoformer_loss

        # optimize SAM and homoformer saperately
        sam_optimizer, homoformer_optimizer = self.optimizers()
        homoformer_scheduler = self.lr_schedulers()
        if self.args.unfreeze_sam_head:
            self.manual_backward(self.sam_backward_loss, retain_graph=True)
            sam_optimizer.step()
            sam_optimizer.zero_grad()

        self.manual_backward(self.homoformer_loss)
        homoformer_optimizer.step()
        homoformer_optimizer.zero_grad()
        if self.trainer.is_last_batch:
            homoformer_scheduler.step()


        self.log('homoformer_loss', self.homoformer_loss.item(), on_step=True)
        self.log('segmentation_loss', self.segmentation_loss.item(), on_step=True)
        self.log('sam_backward_loss', self.sam_backward_loss, on_step=True)
        self.log('laplacian_loss', laplacian_loss.item(), on_step=True)
        self.log('gradient_loss', gradient_ori_loss.item(), on_step=True)
        self.log('soft_mask_loss', self.soft_mask_loss.item(), on_step=True)

        return {'soft_mask': sam_pred_soft_mask,
                'residual_mask': residual_mask}


    def sample(self, data):
        tile = 384
        sam_input = data['sam_SR']
        _, sam_pred_soft_mask = self.sam(sam_input)
        sam_pred_soft_mask = torch.sigmoid(sam_pred_soft_mask)
        data['mask'] = F.interpolate(sam_pred_soft_mask, size=(data['HR'].shape[2], data['HR'].shape[3]), mode='bilinear', align_corners=True)
        # rgb_gt = data['HR'].numpy().squeeze().transpose((1, 2, 0))
        input = data['SR']
        save_image(input, "test_input.png")
        mask = data['mask']
        B, C, H, W = input.shape
        tile_overlap = 30
        split_data, starts = splitimage(input, crop_size=tile, overlap_size=tile_overlap)
        mask_data, starts = splitimage(mask, crop_size=tile, overlap_size=tile_overlap)
        for i, (data, mask_) in enumerate(zip(split_data, mask_data)):
            split_data[i] = self.homoformer(data, mask_)
        restored = mergeimage(split_data, starts, crop_size=tile, resolution=(B, C, H, W))
        # rgb_restored = torch.clamp(restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))
        # shadow_removal_sr, diffusion_mask_pred = self.diffusion.super_resolution(data['SR'], data['mask'], continous=False)
        return restored, sam_pred_soft_mask

    def print_param_values(self):
        print("SAM parameters:")
        for name, param in self.sam.mask_decoder.soft_mask_decoder_adapter.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")
            print(f"{name}: {param.data.mean().item()}")
            if param.grad is not None:
                print(f"{name}: {param.grad.mean().item()}")
            else:
                print(f"{name}: NO gradient")
            break

    def training_step(self, train_data, batch_idx):
        result = self(train_data)
        # loss = result['loss']
        sam_pred_soft_mask = result['soft_mask']
        residual_mask = result['residual_mask']

        # print("After training step:")
        # self.print_param_values()
        if self.global_step % 100 == 0:
            self.logger.experiment.add_image('Train/sam_pred_mask', sam_pred_soft_mask[0], self.global_step)
            self.logger.experiment.add_image('Train/residual_mask', residual_mask[0], self.global_step)

    def validation_step(self, val_data, batch_idx):
        homoformer_sr, sam_pred_soft_mask = self.sample(val_data)
        res = Metrics.tensor2img(homoformer_sr)
        hr_img = Metrics.tensor2img(val_data['HR'])
        eval_psnr = Metrics.calculate_psnr(res, hr_img)
        eval_ssim = Metrics.calculate_ssim(res, hr_img)
        log_img = homoformer_sr[0].clamp_(-1, 1)
        log_img = (log_img + 1) / 2
        self.logger.experiment.add_image('Val/Shadow_Removal', log_img, self.global_step)
        self.logger.experiment.add_image('Val/Diffusion_Mask', sam_pred_soft_mask[0],self.global_step)
        self.log('Val/psnr', eval_psnr, sync_dist=True)
        self.log('Val/ssim', eval_ssim, sync_dist=True)


    def predict_step(self, test_data):
        homoformer_sr, sam_pred_soft_mask = self.sample(test_data)
        res = Metrics.tensor2img(homoformer_sr, min_max=(0, 1))
        hr_img = Metrics.tensor2img(test_data['HR'], min_max=(0, 1))
        eval_psnr = Metrics.calculate_psnr(res, hr_img)
        eval_ssim = Metrics.calculate_ssim(res, hr_img)
        filename = test_data['filename']
        return eval_psnr, eval_ssim, filename, homoformer_sr, sam_pred_soft_mask, test_data['HR']


    def load_pretrained_models(self, path):
        self.homoformer.load_state_dict(torch.load(path.homoformer, map_location=self.device
                                                   ), strict=False)

        sam_state_dict = torch.load(path.sam, map_location=self.device)
        self.sam.load_state_dict(sam_state_dict, strict=False)

    def configure_optimizers(self):
        sam_optimizer = torch.optim.Adam(self.sam.parameters(), lr=0.0002)
        homoformer_optimizer = torch.optim.Adam(self.homoformer.parameters(), lr=self.args.lr_initial,
                                                betas=(0.9, 0.999), eps=1e-8, weight_decay=self.args.weight_decay)
        warmup_epochs = self.args.warmup_epochs
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(homoformer_optimizer,
                                                                      self.args.nepoch - warmup_epochs, eta_min=1e-6)
        homoformer_scheduler = GradualWarmupScheduler(homoformer_optimizer, multiplier=1, total_epoch=warmup_epochs,
                                           after_scheduler=scheduler_cosine)

        return [sam_optimizer, homoformer_optimizer], [homoformer_scheduler]


# using SAM mask train HomoFormer
def train(args: DictConfig) -> None:
    logger = TensorBoardLogger(save_dir=f"./experiments_lightning/{args.name}",
                               name=args.version + "_logger")
    data_module = SamShadowDataModule(args)
    data_module.setup("fit")
    ckpt_path = args.ckpt_path
    model = SamShadow(SAM, HomoFormer, args, ckpt_path)

    # load training parameters
    save_model_name = args.name
    max_epochs = args.max_epochs
    save_every_n_epochs = args.every_n_epochs
    log_every_n_steps = 1

    save_path = f"./experiments_lightning/{save_model_name}/{args.version}"
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        filename='{epoch}',
        save_top_k=-1,
        every_n_epochs=save_every_n_epochs,  # Save every 20 epochs
        save_last=True,  # Save the last model as well
    )
    trainer = L.Trainer(
        accelerator='gpu',
        precision=16,
        devices=args.gpu_ids,
        max_epochs=max_epochs,
        default_root_dir=save_path,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        num_sanity_val_steps=0,
        # check_val_every_n_epoch=10,  #Haven't write val data yet
        log_every_n_steps=log_every_n_steps,
        strategy=DDPStrategy(gradient_as_bucket_view=True)
    )
    trainer.fit(model, data_module)


def sample(args: DictConfig) -> None:
    logger = TensorBoardLogger(save_dir=f"./experiments_lightning/{args.name}", name="sample")
    data_module = SamShadowDataModule(args)
    data_module.setup("test")
    model = SamShadow.load_from_checkpoint(args.samshadow_ckpt_path)
    predictor = L.Trainer(
        accelerator='gpu',
        devices=args.gpu_ids,
        max_epochs=-1,
        benchmark=True,
        logger=logger
    )
    predictions = predictor.predict(model, data_module)
    PSNR_SSIM_list_with_name = []
    PSNR = []
    SSIM = []
    save_path = args.save_result_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i in range(len(predictions)):
        eval_psnr, eval_ssim, filename, homoformer_sr, sam_pred_soft_mask, gt_image = predictions[i]
        filename = filename[0]
        PSNR_SSIM_list_with_name.append((f'{filename}_PSNR', eval_psnr))
        PSNR_SSIM_list_with_name.append((f'{filename}_SSIM', eval_ssim))
        PSNR.append(eval_psnr)
        SSIM.append(eval_ssim)
        # Save SR image
        sr_path = os.path.join(save_path, f'{filename}_sr.png')
        res = Metrics.tensor2img(homoformer_sr, min_max=(0, 1))
        sr_img = Image.fromarray(res)
        sr_img.save(sr_path)
        # Save HR image
        hr_path = os.path.join(save_path, f'{filename}_hr.png')
        hr_img = Metrics.tensor2img(gt_image, min_max=(0, 1))
        hr_img = Image.fromarray(hr_img)
        hr_img.save(hr_path)
        # Save mask
        soft_mask = sam_pred_soft_mask.squeeze().cpu().numpy() * 255
        soft_mask_path = os.path.join(save_path, f'{filename}_soft_mask.png')
        soft_mask_img = Image.fromarray(soft_mask.astype(np.uint8))
        soft_mask_img.save(soft_mask_path)


    psnr_mean = np.mean(PSNR)
    ssim_mean = np.mean(SSIM)
    print(f'PSNR: {psnr_mean:}')
    print(f'SSIM: {ssim_mean:}')

    with open(os.path.join(save_path, 'PSNR_SSIM_list.log'), 'w') as file:
        file.write(f"PSNR_mean: {psnr_mean}\n")
        file.write(f"SSIM_mean: {ssim_mean}\n")
        for key, value in PSNR_SSIM_list_with_name:
            file.write(f"{key}: {value}\n")


@hydra.main(config_path="./config", config_name='soft_mask_homoformer_SRD', version_base=None)
def main(args: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    L.seed_everything(1234)
    args.version = args.get('version', 'none_version')

    if args.phase == 'train':
        train(args)
    elif args.phase == 'test':
        sample(args)


if __name__ == "__main__":
    main()

# change args