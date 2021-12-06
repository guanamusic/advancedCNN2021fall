import torch

from model.base import BaseModule
from model.generator import Generator
from model.discriminator import LocalDiscriminator, GlobalDiscriminator

from tools import crop_only_masked_region

class Inpainter(BaseModule):
    """
    Model (including neural network) works here
    """
    def __init__(self, config):
        super(Inpainter, self).__init__()
        self.config = config
        self.netG = Generator(self.config)
        self.localD = LocalDiscriminator(self.config)
        self.globalD = GlobalDiscriminator(self.config)

    def compute_loss(self, masked_input, mask, ground_truth, spatial_discounting_mask, compute_g_loss=False, compute_d_loss=True):
        """
        Compute the losses
        """
        losses = {}
        l1_loss = torch.nn.L1Loss()

        x_stage1, x_stage2, offset_flow = self.netG(masked_input, mask)
        local_patch_gt = crop_only_masked_region(ground_truth, mask)
        x_inpaint_stage1 = x_stage1 * mask + masked_input * (torch.ones_like(mask) - mask)
        x_inpaint_stage2 = x_stage2 * mask + masked_input * (torch.ones_like(mask) - mask)
        local_patch_stage1 = crop_only_masked_region(x_inpaint_stage1, mask)
        local_patch_stage2 = crop_only_masked_region(x_inpaint_stage2, mask)

        # Discriminator part
        # WGAN D loss
        if compute_d_loss:
            local_patch_real_pred, local_patch_fake_pred = self.dis_forward(
                self.localD, local_patch_gt, local_patch_stage2.detach())
            global_real_pred, global_fake_pred = self.dis_forward(
                self.globalD, ground_truth, x_inpaint_stage2.detach())
            losses['wgan_d'] = torch.mean(local_patch_fake_pred - local_patch_real_pred) + \
                torch.mean(global_fake_pred - global_real_pred) * self.config.model_config.global_wgan_loss_alpha
            # gradients penalty loss
            local_penalty = self.calc_gradient_penalty(
                self.localD, local_patch_gt, local_patch_stage2.detach())
            global_penalty = self.calc_gradient_penalty(self.globalD, ground_truth, x_inpaint_stage2.detach())
            losses['wgan_gp'] = local_penalty + global_penalty

            losses['total_discriminator'] = losses['wgan_d'] + \
                                            losses['wgan_gp'] * self.config.model_config.wgan_gp_lambda

        # Generator part
        if compute_g_loss:
            sd_mask = spatial_discounting_mask
            losses['l1'] = l1_loss(local_patch_stage1 * sd_mask, local_patch_gt * sd_mask) * \
                self.config.model_config.coarse_l1_alpha + \
                l1_loss(local_patch_stage2 * sd_mask, local_patch_gt * sd_mask)
            losses['ae'] = l1_loss(x_stage1 * (1. - mask), ground_truth * (1. - mask)) * \
                self.config.model_config.coarse_l1_alpha + \
                l1_loss(x_stage2 * (1. - mask), ground_truth * (1. - mask))
            # WGAN G loss
            local_patch_real_pred, local_patch_fake_pred = self.dis_forward(
                self.localD, local_patch_gt, local_patch_stage2)
            global_real_pred, global_fake_pred = self.dis_forward(
                self.globalD, ground_truth, x_stage2)
            losses['wgan_g'] = - torch.mean(local_patch_fake_pred) - \
                torch.mean(global_fake_pred) * self.config.model_config.global_wgan_loss_alpha

            losses['total_generator'] = losses['l1'] * self.config.model_config.l1_loss_alpha \
                                        + losses['ae'] * self.config.model_config.ae_loss_alpha \
                                        + losses['wgan_g'] * self.config.model_config.gan_loss_alpha

        return losses

    def dis_forward(self, netD, ground_truth, x_inpaint):
        assert ground_truth.size() == x_inpaint.size()
        batch_size = ground_truth.size(0)
        batch_data = torch.cat([ground_truth, x_inpaint], dim=0)
        batch_output = netD(batch_data)
        real_pred, fake_pred = torch.split(batch_output, batch_size, dim=0)
        return real_pred, fake_pred

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data).to(real_data.device)

        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.requires_grad_().clone()

        disc_interpolates = netD(interpolates)
        grad_outputs = torch.ones(disc_interpolates.size()).to(real_data.device)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=grad_outputs, create_graph=True,
                                        retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def forward(self, masked_input, mask):
        """
        Compute forward pass of neural network
        """
        # actually, IDK lmao
        _, inpainted, _ = self.netG(masked_input, mask)
        return inpainted