from __future__ import print_function
import argparse
import cv2
import torch
import process_stylization
from photo_wct import PhotoWCT
import numpy as np
import json
parser = argparse.ArgumentParser(description='Photorealistic Image Stylization')
parser.add_argument('--model', default='./PhotoWCTModels/photo_wct.pth')
parser.add_argument('--content_video_path', default='./images/content1.jpg')
parser.add_argument('--content_seg_path', default=[])
parser.add_argument('--style_image_path', default='./images/style1.jpg')
parser.add_argument('--style_seg_path', default=[])
parser.add_argument('--output_image_path', default='./results/example1.jpg')
parser.add_argument('--save_intermediate', action='store_true', default=False)
parser.add_argument('--fast', action='store_true', default=False)
parser.add_argument('--no_post', action='store_true', default=True)
parser.add_argument('--cuda', type=int, default=0, help='Enable CUDA.')
args = parser.parse_args()


# Load model
p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load(args.model))

if args.fast:
    from photo_gif import GIFSmoothing
    p_pro = GIFSmoothing(r=35, eps=0.001)
else:
    from photo_smooth import Propagator
    p_pro = Propagator()
if args.cuda:
    p_wct.cuda(0)

cap = cv2.VideoCapture(args.content_video_path)
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height =int(cap.get( cv2.CAP_PROP_FRAME_HEIGHT)) 
out = cv2.VideoWriter('output_test.avi', fourcc, fps, (frame_width, frame_height)) 
while(True) :
    ret, frame = cap.read()
    if(ret == False) :
        break

    cv2.imwrite('frame.jpg', frame)
    process_stylization.stylization(
            stylization_module=p_wct,
            smoothing_module=p_pro,
            content_image_path='frame.jpg',
            style_image_path=args.style_image_path,
            content_seg_path=args.content_seg_path,
            style_seg_path=args.style_seg_path,
            output_image_path='output.jpg',
            cuda=args.cuda,
            save_intermediate=args.save_intermediate,
            no_post=args.no_post
        )

    stylised_frame = cv2.imread('output.jpg')
    stylised_frame = cv2.resize(stylised_frame, (frame_width, frame_height), interpolation = cv2.INTER_AREA) 
    out.write(stylised_frame)

cap.release()
out.release()
cv2.destroyAllWindows()