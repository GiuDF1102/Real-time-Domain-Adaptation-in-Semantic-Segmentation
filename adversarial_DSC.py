#!/usr/bin/python
# -*- encoding: utf-8 -*-
from model.model_stages import BiSeNet
from cityscapes import CityScapes
from gta5 import GTA5
import torch
from torch.utils.data import DataLoader
import logging
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch.cuda.amp as amp
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from tqdm import tqdm
import matplotlib.pyplot as plt

#FOR ADVERSARIAL
import torch.nn.functional as F
from model.discriminator_dsc import FCDiscriminator



logger = logging.getLogger()


def val(args, model, dataloader):
    print('start val!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict, _, _ = model(data)
            predict = predict.squeeze(0)
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)

        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print(f'mIoU per class: {miou_list}')

        return precision, miou

def train(args, model, optimizer, dataloader_train, dataloader_val):
    writer = SummaryWriter(comment=''.format(args.optimizer))

    scaler = amp.GradScaler()

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    max_miou = 0
    step = 0
    miou_list = []
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i, (data, label) in enumerate(dataloader_train):
            data = data.cuda()
            label = label.long().cuda()
            optimizer.zero_grad()

            with amp.autocast():
                output, out16, out32 = model(data)
                loss1 = loss_func(output, label.squeeze(1))
                loss2 = loss_func(out16, label.squeeze(1))
                loss3 = loss_func(out32, label.squeeze(1))
                loss = loss1 + loss2 + loss3

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest.pth'))

        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val)
            miou_list.append(miou)
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)
    plt.plot(range(args.num_epochs), miou_list)
    plt.xlabel("Epoch #")
    plt.ylabel("mIoU")
    plt.savefig(os.path.join("/content/drive/MyDrive/figures",args.figure_name))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--mode',
                       dest='mode',
                       type=str,
                       default='train',
    )
    parse.add_argument('--augmentation',
                       dest='augmentation',
                       type=str,
                       default='H',
    )
    parse.add_argument('--backbone',
                       dest='backbone',
                       type=str,
                       default='STDCNet813',
    )
    parse.add_argument('--pretrain_path',
                      dest='pretrain_path',
                      type=str,
                      default='./STDCNet813M_73.91.tar',
    )
    parse.add_argument('--use_conv_last',
                       dest='use_conv_last',
                       type=str2bool,
                       default=False,
    )
    parse.add_argument('--num_epochs',
                       type=int, default=50,
                       help='Number of epochs to train for')
    parse.add_argument('--epoch_start_i',
                       type=int,
                       default=0,
                       help='Start counting epochs from this number')
    parse.add_argument('--checkpoint_step',
                       type=int,
                       default=1,
                       help='How often to save checkpoints (epochs)')
    parse.add_argument('--validation_step',
                       type=int,
                       default=49,
                       help='How often to perform validation (epochs)')
    parse.add_argument('--batch_size',
                       type=int,
                       default=8,
                       help='Number of images in each batch')
    parse.add_argument('--learning_rate',
                        type=float,
                        default=0.01,
                        help='learning rate used for train')
    parse.add_argument('--learning_rate_discriminator',
                        type=float,
                        default=0.0001,
                        help='learning rate used for train discriminator')
    parse.add_argument('--lambda_adv',
                        type=float,
                        default=0.001,
                        help='lambda for adversarial')
    parse.add_argument('--num_workers',
                       type=int,
                       default=4,
                       help='num of workers')
    parse.add_argument('--num_classes',
                       type=int,
                       default=19,
                       help='num of object classes (with void)')
    parse.add_argument('--cuda',
                       type=str,
                       default='0',
                       help='GPU ids used for training')
    parse.add_argument('--use_gpu',
                       type=bool,
                       default=True,
                       help='whether to user gpu for training')
    parse.add_argument('--save_model_path',
                       type=str,
                       default=None,
                       help='path to save model')
    parse.add_argument('--optimizer',
                       type=str,
                       default='sgd',
                       help='optimizer, support rmsprop, sgd, adam')
    parse.add_argument('--loss',
                       type=str,
                       default='crossentropy',
                       help='loss function')


    return parse.parse_args()


def train_adversarial(args, lambda_adv, model, model_D, optimizer, optimizer_discriminator, dataloader_target, dataloader_source, dataloader_val):
    """
    This framework is based on a Generative Adversarial Network (GAN). It is composed by:
     - A segmentation model G to predict output results.
     - A Discriminator D to distinguish whether the input is from the source or target domain.
    """
    # For logging the optimization of the objective function.
    writer = SummaryWriter(comment=''.format(args.optimizer))

    # Sometimes gradients are too small to be taken into account, GradScaler scales them by some factor (NOT VANISHING GRADIENTS)
    # Simply: float16-32 tensors often don't take into account extremely small variations.
    scaler = amp.GradScaler()

    # Loss function used, specifing that class 255 should be ignored.
    # Pretty much, whatever this model outputs a pixel of class 255 will be ignored by the loss function. 
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    bce_loss = torch.nn.BCEWithLogitsLoss()

    # All this is for logging maximum mIoU, epoch step, mIoU list
    max_miou = 0
    step = 0
    miou_list = []

    # Labels for adversarial training
    adv_source_label = 0
    adv_target_label = 1


    # Training loop
    for epoch in range(args.num_epochs):
        # In pytorch a model is updated by an optimizer and the learning rate schedule is an algorithm to update
        # the learning rate in an optimizer. This function applies a decay of the learning rate. Learning rate decay 
        # involves gradually reducing the learning rate over time during training. 
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        lr_discriminator = poly_lr_scheduler(optimizer_discriminator, args.learning_rate_discriminator, iter=epoch, max_iter=args.num_epochs)
        
        # Tells your model that you are training the model. This helps inform layers such as Dropout and BatchNorm, 
        # which are designed to behave differently during training and evaluation.
        model.train()
        model_D.train()

        # tqdm settings, numbers of batches times batch size, to track training. Then a description with the output
        tq = tqdm(total=len(dataloader_source) * args.batch_size)
        tq.set_description('epoch %d, lr %f, lr_d %f' % (epoch, lr, lr_discriminator))

        # vector to log the loss
        loss_record = []

        # Training loop
        for ((data_source, label_source), (data_target, _)) in zip(dataloader_source, dataloader_target):
            # image and label are being moved to the GPU
            data_source = data_source.cuda()
            data_target = data_target.cuda()
            label_source = label_source.long().cuda()

            # For every mini-batch during the training phase, we set the gradients to zero before starting to do 
            # backpropagation. Otherwise, the gradient would be a combination of the old gradient, which you have 
            # already used to update your model parameters and the newly-computed gradient.
            optimizer.zero_grad()
            optimizer_discriminator.zero_grad()

            # don't accumulate grads in D
            for param in model_D.parameters():
                param.requires_grad = False


            """
                We first forward the source image Is (with annotations) to the segmentation network for optimizing G.
            """
            # Instances of autocast serve as context managers or decorators that allow regions of your script to 
            # run in mixed precision.
            with amp.autocast():
                # Get model outputs, "we use the Stage 3, 4, 5 to produce the feature maps with
                # down-sample ratio 1/8, 1/16, 1/32"
                output_source, out16_source, out32_source = model(data_source)
                output_target, _, _ = model(data_target)
                # Apply loss
                loss1_source = loss_func(output_source, label_source.squeeze(1))
                loss2_source = loss_func(out16_source, label_source.squeeze(1))
                loss3_source = loss_func(out32_source, label_source.squeeze(1))
                # Combine loss
                loss_source = loss1_source + loss2_source + loss3_source

            scaler.scale(loss_source).backward()

            """
                Then we predict the segmentation softmax output Pt for the target image It (without annotations).
            """
            with amp.autocast():
                """
                    Since our goal is to make segmentation predictions P of source and target images (i.e., Ps and Pt) 
                    close to each other, we use these two predictions as the input to the discriminator Di to distinguish 
                    whether the input is from the source or target domain
                """
                output_discriminator = model_D(F.softmax(output_target, dim=1))

                """
                    With an adversarial loss on the target prediction, the network propagates gradients from Di
                    to G, which would encourage G to generate similar segmentation distributions in
                    the target domain to the source prediction
                """
                loss_adv_target = bce_loss(output_discriminator, torch.FloatTensor(output_discriminator.data.size()).fill_(adv_source_label).cuda())

            # Scales the loss and does a backward pass
            scaler.scale(lambda_adv*loss_adv_target).backward()

            """
                Here we train the discriminator to distinguish between target and source
            """
            # bring back requires_grad
            for param in model_D.parameters():
                param.requires_grad = True

            # train with source
            output_source = output_source.detach()
            
            with amp.autocast():
                output_discriminator_train = model_D(F.softmax(output_source, dim=1))

                loss_discriminator = bce_loss(output_discriminator_train, torch.FloatTensor(output_discriminator_train.data.size()).fill_(adv_source_label).cuda())
            loss_discriminator = loss_discriminator / 2
            scaler.scale(loss_discriminator).backward()

            # train with target
            output_target = output_target.detach()
            
            with amp.autocast():
                output_discriminator_train = model_D(F.softmax(output_target, dim=1))

            loss_discriminator = bce_loss(output_discriminator_train, torch.FloatTensor(output_discriminator_train.data.size()).fill_(adv_target_label).cuda())
            loss_discriminator = loss_discriminator / 2
            scaler.scale(loss_discriminator).backward()

            scaler.step(optimizer)
            scaler.step(optimizer_discriminator)
            scaler.update()

            # tqd, updates for each batch
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss_source)
            step += 1
            writer.add_scalar('loss_step', loss_source, step)
            loss_record.append(loss_source.item())
        
        # Training ended
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest.pth'))
            torch.save(model_D.module.state_dict(), os.path.join(args.save_model_path, 'latest_discriminator.pth'))

        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val)
            miou_list.append(miou)
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best.pth'))
                torch.save(model_D.module.state_dict(), os.path.join(args.save_model_path, 'best_discriminator.pth'))

            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)
    plt.plot(range(args.num_epochs), miou_list)
    plt.xlabel("Epoch #")
    plt.ylabel("mIoU")
    plt.savefig(os.path.join("/content/drive/MyDrive/figures",args.figure_name))

def main():
    args = parse_args()

    ## dataset
    n_classes = args.num_classes

    mode = args.mode

    source_dataset = GTA5(mode='train_full', aug_type=args.augmentation)
    target_dataset = CityScapes(mode='train')
    val_dataset = CityScapes(mode='val')

    # dataloader
    dataloader_source = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)
    dataloader_target = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    dataloader_val = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False)

    model_discriminator = FCDiscriminator(num_classes=args.num_classes)

    if(args.pretrain_path == './STDCNet813M_73.91.tar'):
        model = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, use_conv_last=args.use_conv_last)
    else:
        model_ckpt = args.pretrain_path+'/best.pth'
        modeldiscriminator_ckpt = args.pretrain_path+'/latest_discriminator.pth'
        model = BiSeNet(backbone=args.backbone, n_classes=n_classes, use_conv_last=args.use_conv_last)
        model.load_state_dict(torch.load(model_ckpt), strict=True)
        model_discriminator.load_state_dict(torch.load(modeldiscriminator_ckpt), strict=True)


    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
        model_discriminator = torch.nn.DataParallel(model_discriminator).cuda()

    ## optimizer
    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    optimizer_discriminator = torch.optim.Adam(model_discriminator.parameters(), lr=args.learning_rate_discriminator, betas=(0.9, 0.99))

    #train
    train_adversarial(args, args.lambda_adv, model, model_discriminator, optimizer, optimizer_discriminator, dataloader_target, dataloader_source, dataloader_val)

    # final test
    val(args, model, dataloader_val)

if __name__ == "__main__":
    main()