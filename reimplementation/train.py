import argparse
import os

import torch
from torch.utils.data import DataLoader

from dataset import IlluminationDataset
from losses import GANLoss, feature_loss, image_and_light_loss
from models.discriminator import Discriminator
from models.hourglass import HourglassNet
from utils import check_folder

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(args):

    # check if results path exists, if not create the folder
    check_folder(args.results_path)

    # generator model
    generator = HourglassNet(high_res=args.high_resolution)
    generator.to(device)

    # discriminator model
    discriminator = Discriminator(input_nc=1)
    discriminator.to(device)

    # optimizer
    optimizer_g = torch.optim.Adam(generator.parameters())
    optimizer_d = torch.optim.Adam(discriminator.parameters())

    # training parameters
    feature_weight = 0.5
    skip_count = 0
    use_gan = args.use_gan
    print_frequency = 5

    # dataloader
    illum_dataset = IlluminationDataset()
    illum_dataloader = DataLoader(illum_dataset, batch_size=args.batch_size)

    # gan loss based on lsgan that uses squared error
    gan_loss = GANLoss(gan_mode='lsgan')

    # training
    for epoch in range(1, args.epochs + 1):

        for data_idx, data in enumerate(illum_dataloader):
            source_img, source_light, target_img, target_light = data

            source_img.to(device)
            source_light.to(device)
            target_img.to(device)
            target_light.to(device)

            optimizer_g.zero_grad()

            # if skip connections are required for training, else skip the
            # connections based on the the training scheme for low-res/high-res
            # images
            if args.use_skip:
                skip_count = 0
            else:
                skip_count = 5 if args.high_resolution else 4

            output = generator(
                source_img,
                target_light,
                skip_count,
                target_img)

            source_face_feats, source_light_pred, target_face_feats, source_relit_pred = output

            img_loss = image_and_light_loss(
                source_relit_pred,
                target_img,
                source_light_pred,
                target_light)
            feat_loss = feature_loss(source_face_feats, target_face_feats)

            # if gan loss is used
            if use_gan:
                g_loss = gan_loss(
                    discriminator(source_relit_pred),
                    target_is_real=True)
            else:
                g_loss = torch.Tensor([0])

            total_g_loss = img_loss + g_loss + (feature_weight * feat_loss)
            total_g_loss.backward()
            optimizer_g.step()

            # training the discriminator
            if use_gan:
                optimizer_d.zero_grad()
                pred_real = discriminator(target_img)
                pred_fake = discriminator(source_relit_pred.detach())

                loss_real = gan_loss(pred_real, target_is_real=True)
                loss_fake = gan_loss(pred_fake, target_is_real=False)

                d_loss = (loss_real + loss_fake) * 0.5
                d_loss.backward()
                optimizer_d.step()
            else:
                loss_real = torch.Tensor([0])
                loss_fake = torch.Tensor([0])

            if data_idx % print_frequency == 0:
                print(
                    "Epoch: [{}]/[{}], Iteration: [{}]/[{}], image loss: {}, feature loss: {}, gen fake loss: {}, dis real loss: {}, dis fake loss: {}".format(
                        epoch,
                        args.epochs +
                        1,
                        data_idx +
                        1,
                        len(illum_dataloader),
                        img_loss.item(),
                        feat_loss.item(),
                        g_loss.item(),
                        loss_real.item(),
                        loss_fake.item()))

        # saving model
        checkpoint_path = os.path.join(
            args.results_path,
            'checkpoint_epoch_{}.pth'.format(epoch))
        checkpoint = {
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'optimizer_d': optimizer_d.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)

        # TODO: visualization step


def validation():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training parameters
    parser.add_argument(
        '--epochs',
        '-ep',
        help='Number of epochs to train the model',
        type=int)
    parser.add_argument(
        '--high_resolution',
        '-hr',
        help='If fine-tuning on high-resolution data',
        type=eval,
        choices=[True, False],
        default='False')
    parser.add_argument(
        '--use_skip',
        '-us',
        help='Boolean flag to use skip connections',
        type=eval,
        choices=[True, False],
        default='True')
    parser.add_argument(
        '--use_gan',
        '-ug',
        help='Boolean flag to use gan loss',
        type=eval,
        choices=[True, False],
        default='False')
    parser.add_argument(
        '--batch_size',
        '-bs',
        help='Batch size of the data',
        type=int)

    # results
    parser.add_argument(
        '--results_path',
        '-rp',
        help='Path of the results',
        type=str)

    args = parser.parse_args()

    train(args)
