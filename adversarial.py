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

from model.discriminator import FCDiscriminator
import torch.nn.functional as F

logger = logging.getLogger()

"""
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))
"""
"""
def adjust_learning_rate_D(optimizer, i_iter, lrate, num_steps, power):
    lr = lr_poly(lrate, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
"""

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
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)

def train_adversarial(args, model, model_D1, optimizer, optimizer_D1, dataloader_train, dataloader_target, dataloader_val, iter_size):
    # writer = SummaryWriter(comment=''.format(args.optimizer))

    scaler = amp.GradScaler()

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    loss_bce = torch.nn.BCEWithLogitsLoss()

    source_label = 0
    target_label = 1

    max_miou = 0
    miou_list = []
    step = 0
    for epoch in range(args.num_epochs):
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)

        model.train()
        model_D1.train()

        tq = tqdm(total=(len(dataloader_target) + len (dataloader_train)) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr_D1))
        loss_record = []

        for (data, label), (data_t, _) in zip(dataloader_train, dataloader_target):
            data = data.cuda()
            data_t = data_t.cuda()
            label = label.long().cuda()

            optimizer.zero_grad()
            optimizer_D1.zero_grad()

            # train G
            # don't accumulte grads in D
            for param in model_D1.parameters():
                param.requires_grad = False
            # train with source
            with amp.autocast():
                output, _ , _  = model(data)
                pred_target1, _ , _  = model(data_t)                    
                
                loss1 = loss_func(output, label.squeeze(1))
                # loss2 = loss_func(out16, label.squeeze(1))
                # loss3 = loss_func(out32, label.squeeze(1))
                loss = loss1 # + loss2 + loss3

            scaler.scale(loss).backward()

            with amp.autocast():
                D_out1 = model_D1(F.softmax(pred_target1, dim=1))

                loss_adv_target1 = loss_bce(D_out1, 
                                            torch.FloatTensor(D_out1.data.size())
                                            .fill_(source_label)
                                            .cuda())
                
            loss_adv_target = args.lambda_adv_target1 * loss_adv_target1 
            
            scaler.scale(loss_adv_target).backward()
            loss_adv_target_value1 += loss_adv_target1.data.cpu().numpy()

            # train D
            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True

            # train with source
            output = output.detach()

            with amp.autocast():
                D_out1 = model_D1(F.softmax(output, dim=1))

                loss_D1_source = loss_bce(D_out1, 
                                            torch.FloatTensor(D_out1.data.size())
                                            .fill_(source_label)
                                            .cuda())

            scaler.scale(loss_D1_source).backward()  

            loss_D_value1 += loss_D1_source.data.cpu().numpy()

            # train with target
            pred_target1 = pred_target1.detach()

            with amp.autocast():
                D_out1 = model_D1(F.softmax(pred_target1, dim=1))

                loss_D1_target = loss_bce(D_out1, 
                                            torch.FloatTensor(D_out1.data.size())
                                            .fill_(target_label)
                                            .cuda())
            
            scaler.scale(loss_D1_target).backward()

            scaler.step(optimizer)
            scaler.step(optimizer_D1)
            scaler.update()

            tq.update(args.batch_size*2)
            tq.set_postfix(loss='%.6f' % loss)
            step += 1
            # writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        # writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest.pth'))
            torch.save(model_D1.module.state_dict(), os.path.join(args.save_model_path, 'latest_D1.pth'))

        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val)
            miou_list.append(miou)
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best.pth'))
                torch.save(model_D1.module.state_dict(), os.path.join(args.save_model_path, 'best_D1.pth'))
            # writer.add_scalar('epoch/precision_val', precision, epoch)
            # writer.add_scalar('epoch/miou val', miou, epoch)
    plt.plot(range(args.num_epochs), miou_list)
    plt.xlabel("Epoch #")
    plt.ylabel("mIoU")
    plt.savefig(os.path.join("/content/drive/MyDrive/AML_Project/figures",args.figure_name))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--lambda_adv_target1',
                       dest='lambda_adv_target1',
                       type=float,
                       default=0.001,
    )
    parse.add_argument('--lambda_adv_target2',
                       dest='lambda_adv_target2',
                       type=float,
                       default=0.0001,
    )
    parse.add_argument('--lambda_adv_target3',
                       dest='lambda_adv_target3',
                       type=float,
                       default=0.0001,
    )
    parse.add_argument('--iter_size',
                       dest='iter_size',
                       type=int,
                       default=1,
    )
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
    parse.add_argument('--pretrain_path_D1',
                      dest='pretrain_path_D1',
                      type=str,
                      default=None,
    )
    parse.add_argument('--pretrain_path_D2',
                      dest='pretrain_path_D2',
                      type=str,
                      default=None,
    )
    parse.add_argument('--pretrain_path_D3',
                      dest='pretrain_path_D3',
                      type=str,
                      default=None,
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
                       default=1,
                       help='How often to perform validation (epochs)')
    parse.add_argument('--batch_size',
                       type=int,
                       default=4,
                       help='Number of images in each batch')
    parse.add_argument('--learning_rate',
                        type=float,
                        default=0.01,
                        help='learning rate used for train')
    parse.add_argument('--learning_rate_D',
                        type=float,
                        default=0.0002,
                        help='learning rate used for train discriminator')
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


def main():
    args = parse_args()

    ## dataset
    n_classes = args.num_classes
    iter_size = args.iter_size

    train_dataset = GTA5(mode='train_full', aug_type=args.augmentation)
    target_dataset = CityScapes(mode='train')
    val_dataset = CityScapes(mode='val')

    # dataloader
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)
    dataloader_target = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    dataloader_val = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False)

    if args.pretrain_path != './checkpoints/STDCNet813M_73.91.tar' and args.pretrain_path_D1 is not None:
        print("Starting training with checkpoints " + args.pretrain_path + " " + args.pretrain_path_D1)
        model = BiSeNet(backbone=args.backbone, n_classes=n_classes, use_conv_last=args.use_conv_last)
        model.load_state_dict(torch.load(args.pretrain_path), strict=False)
        model_D1 = FCDiscriminator(num_classes=n_classes, pretrain_model=args.pretrain_path_D1)
        model_D1.load_state_dict(torch.load(args.pretrain_path_D1), strict=False)
    else:        
        model = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, use_conv_last=args.use_conv_last)
        model_D1 = FCDiscriminator(num_classes=n_classes)

    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
        model_D1 = torch.nn.DataParallel(model_D1).cuda()

    ## optimizer
        
    optimizer_D1 = torch.optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    
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

    #train
    train_adversarial(args, 
                      model, 
                      model_D1, 
                      optimizer, 
                      optimizer_D1, 
                      dataloader_train, 
                      dataloader_target, 
                      dataloader_val,
                      iter_size
                      )

    # final test
    val(args, model, dataloader_val)

if __name__ == "__main__":
    main()