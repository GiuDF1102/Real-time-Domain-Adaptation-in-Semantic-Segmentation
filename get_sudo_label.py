from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
from model.model_stages import BiSeNet
import argparse
from cityscapes import CityScapes
from torch.utils.data import DataLoader

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--batch_size',
                       type=int,
                       default=1,
                       help='Number of images in each batch')
    parse.add_argument('--cuda',
                       type=str,
                       default='0',
                       help='GPU ids used for training')
    parse.add_argument('--use_gpu',
                       type=bool,
                       default=True,
                       help='whether to user gpu for training')
    parse.add_argument('--save_sudo',
                       type=str,
                       default='./data/sudo_labels',
                       help='pseudo labels save path')
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

    target_dataset = CityScapes(mode='ssl')
    dataloader_target = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    predicted_label = np.zeros((len(dataloader_target), 512, 1024))
    predicted_prob = np.zeros((len(dataloader_target), 512, 1024))
    image_name = []

    with torch.no_grad():
        for index, batch in enumerate(dataloader_target):
            if index % 100 == 0:
                print( '%d processd' % index )
            image, _, name = batch
            image = image.cuda()

            # forward
            output1, _, _ = model1(image)
            output1 = nn.functional.softmax(output1, dim=1)
            output2, _, _ = model2(image)
            output2 = nn.functional.softmax(output2, dim=1)
            output3, _, _ = model2(image)
            output3 = nn.functional.softmax(output2, dim=1)

            a, b = 0.3333, 0.3333
            output = a*output1 + b*output2 + (1.0-a-b)*output3
            output = np.array(output.squeeze(0).cpu())
            output = output.transpose(1,2,0)
       
            label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
            predicted_label[index] = label.copy()
            predicted_prob[index] = prob.copy()
            image_name.append(name[0])
        
    thres = []
    for i in range(19):
        x = predicted_prob[predicted_label==i]
        if len(x) == 0:
            thres.append(0)
            continue        
        x = np.sort(x)
        thres.append(x[np.int(np.round(len(x)*0.66))])
    print( thres )
    thres = np.array(thres)
    thres[thres>0.9]=0.9
    print( thres )

    for index in range(len(dataloader_target)):
        name = image_name[index]
        label = predicted_label[index]
        prob = predicted_prob[index]
        for i in range(19):
            label[   (prob<thres[i]) * (label==i)   ] = 255  
        output = np.asarray(label, dtype=np.uint8)
        output = Image.fromarray(output)
        filename = name.split("\\")[-1]
        output.save('%s/%s' % (args.save_sudo, filename)) 
    
if __name__ == '__main__':
    main()
    