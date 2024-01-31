import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.model_stages import BiSeNet
from cityscapes import CityScapes
from utils import fast_hist, per_class_iu, compute_global_accuracy, reverse_one_hot, colour_code_segmentation

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parse_args():
    parse = argparse.ArgumentParser()

    parse.add_argument('--num_classes',
                       type=int,
                       default=19,
                       help='num of object classes (with void)')
    parse.add_argument('--num_workers',
                       type=int,
                       default=4,
                       help='num of workers')
    parse.add_argument('--use_conv_last',
                       dest='use_conv_last',
                       type=str2bool,
                       default=False,
    )
    parse.add_argument('--backbone',
                       dest='backbone',
                       type=str,
                       default='CatmodelSmall',
    )

    return parse.parse_args()

def val_multi(model1, model2, model3, targetloader, args):
    print('start val!')
    # compute scores and save them
    with torch.no_grad():
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for image, label in tqdm(targetloader):
            # forward
            image = image.cuda()
            label = label.type(torch.LongTensor)
            label = label.long().cuda()

            output1, _, _ = model1(image)
            output1 = nn.functional.softmax(output1, dim=1)

            output2, _, _ = model2(image)
            output2 = nn.functional.softmax(output2, dim=1)

            output3, _, _ = model1(image)
            output3 = nn.functional.softmax(output1, dim=1)

            a, b = 0.3333, 0.3333
            predict = a*output1 + b*output2 + (1.0-a-b)*output3
            predict = predict.squeeze(0)
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            precision_record.append(precision)

        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print(f'mIoU per class: {miou_list}')



def main():
    args = parse_args()

    model1 = BiSeNet(backbone=args.backbone, n_classes=args.num_classes, use_conv_last=args.use_conv_last)
    model1.load_state_dict(torch.load('./ADV_v3_DSC_FDA/latest.pth'))
    model2 = BiSeNet(backbone=args.backbone, n_classes=args.num_classes, use_conv_last=args.use_conv_last)
    model2.load_state_dict(torch.load('./ADV_v3_DSC_FDA_005/latest.pth'))
    model3 = BiSeNet(backbone=args.backbone, n_classes=args.num_classes, use_conv_last=args.use_conv_last)
    model3.load_state_dict(torch.load('./ADV_v3_DSC_FDA_009/latest.pth'))

    model1.eval()
    model1.cuda()
    model2.eval()
    model2.cuda()
    model3.eval()
    model3.cuda()

    val_dataset = CityScapes(mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False)

    val_multi(model1, model2, model3, val_dataloader, args)


if __name__ == '__main__':
    main()