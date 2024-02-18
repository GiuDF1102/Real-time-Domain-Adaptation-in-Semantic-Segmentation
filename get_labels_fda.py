import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.model_stages import BiSeNet
from cityscapes import CityScapes
from utils import fast_hist, per_class_iu, compute_global_accuracy, reverse_one_hot, colour_code_segmentation
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image

translation = [[128,64,128],
    [244,35,232],
    [70,70,70],
    [102,102,156],
    [190,153,153],
    [153,153,153],
    [250,170,30],
    [220,220,0],
    [107,142,35],
    [152,251,152],
    [70,130,180],
    [220,20,60],
    [255,0,0],
    [0,0,142],
    [0,0,70],
    [0,60,100],
    [0,80,100],
    [0,0,230],
    [119,11,32],
    [0,0,0]]


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
    parse.add_argument('--image_to_print',
                       dest='image_to_print',
                       type=int,
    )

    return parse.parse_args()

def val_multi(args, model1, model2, model3, targetloader):
    print('start val!')
    # compute scores and save them
    with torch.no_grad():
        precision_record = []
        for i, (image, label) in enumerate(targetloader):
            # forward
            # image = image.cuda()
            label = label.type(torch.LongTensor)
            # label = label.long().cuda()

            output1, _, _ = model1(image)
            output1 = nn.functional.softmax(output1, dim=1)

            output2, _, _ = model2(image)
            output2 = nn.functional.softmax(output2, dim=1)

            output3, _, _ = model1(image)
            output3 = nn.functional.softmax(output1, dim=1)

            a, b = 0.3333, 0.3333
            predict1 = a*output1 + b*output2 + (1.0-a-b)*output3

            # get RGB label image
            label = label.squeeze()
            label = np.array(label.cpu())
          
            # Convert prediction to numpy array
            predict1 = predict1.cpu().numpy()

            # Assuming predict contains class probabilities, find the class index with the highest probability
            predicted_labels = np.argmax(predict1, axis=1)

            # In your main function or where you call val() function
            # Call save_segmentation_mask() to save the predicted segmentation mask

            if i == args.image_to_print:
              print("generating image")
              for j in range(len(predicted_labels)):
                  save_segmentation_mask(predicted_labels[j], f"./segmentation_maps/segmentation_mask_{i}_DSC_FDA.png")          

# Function to save the predicted segmentation mask as an image
def save_segmentation_mask(segmentation_mask, filename):
    # Create a color mask for visualization using the translation dictionary
    color_mask = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in enumerate(translation):
        color_mask[segmentation_mask == class_idx] = color

    # Convert the color mask array to a PIL image
    pil_image = Image.fromarray(color_mask)

    # Save the image
    pil_image.save(filename)


def main():
    args = parse_args()

    model1 = BiSeNet(backbone=args.backbone, n_classes=args.num_classes, use_conv_last=args.use_conv_last)
    model1.load_state_dict(torch.load(f'./ADV_v3_DSC_FDA/latest.pth', map_location=torch.device('cpu')))
    model2 = BiSeNet(backbone=args.backbone, n_classes=args.num_classes, use_conv_last=args.use_conv_last)
    model2.load_state_dict(torch.load(f'./ADV_v3_DSC_FDA_005/latest.pth', map_location=torch.device('cpu')))
    model3 = BiSeNet(backbone=args.backbone, n_classes=args.num_classes, use_conv_last=args.use_conv_last)
    model3.load_state_dict(torch.load(f'./ADV_v3_DSC_FDA_009/latest.pth', map_location=torch.device('cpu')))

    model1.eval()
    model2.eval()
    model3.eval()

    val_dataset = CityScapes(mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False)
    val_multi(args, model1, model2, model3, val_dataloader)


if __name__ == '__main__':
    main()