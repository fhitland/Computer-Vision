import torchvision
import torch
import argparse
import segmentation_utils
import cv2

from PIL import Image

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input image')
args = vars(parser.parse_args())

# set computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# download or load the model from disk
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
# model to eval() model and load onto computation devicce
model.eval().to(device)

# read the image
image = Image.open(args['input'])
# do forward pass and get the output dictionary
outputs = segmentation_utils.get_segment_labels(image, model, device)
# get the data from the `out` key
outputs = outputs['out']
segmented_image = segmentation_utils.draw_segmentation_map(outputs)

final_image = segmentation_utils.image_overlay(image, segmented_image)
save_name = f"{args['input'].split('/')[-1].split('.')[0]}"
# show the segmented image and save to disk
cv2.imshow('Segmented image', final_image)
cv2.waitKey(0)
cv2.imwrite(f"outputs/{save_name}.jpg", final_image)