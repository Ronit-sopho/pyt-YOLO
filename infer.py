import numpy as np
import torch
import torch.nn as nn
import gc
from PIL import Image
from model.net import Yolov3Loss
from torchvision.ops import nms
from utils import *
import argparse

torch.set_printoptions(precision=10)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_bounding_boxes(yolo_output, net_size, img_size):
	boxes = []
	scores = []
	net_w, net_h = net_size
	w_factor = img_size[0]/net_w
	h_factor = img_size[1]/net_h
	x_offset = img_size[0]-net_w
	y_offset = img_size[1]-net_h

	xx = yolo_output[...,0]*net_w*w_factor # centre x
	yy = yolo_output[...,1]*net_h*h_factor  # centre y
	ww = yolo_output[...,2]*net_w*w_factor
	hh = yolo_output[...,3]*net_h*h_factor

	# convert to x1, y1, x2, y2 format
	xx1 = xx-(ww/2)
	xx2 = xx+(ww/2)
	yy1 = yy-(hh/2)
	yy2 = yy+(hh/2)

	obj_scores = yolo_output[...,4].unsqueeze_(-1)
	class_probs = yolo_output[...,5:]
	box_scores = obj_scores*class_probs
	boxed = [xx1.flatten(), yy1.flatten(), xx2.flatten(), yy2.flatten()]
	boxes = torch.transpose(torch.stack(boxed),0,1)

	return boxes, box_scores.flatten(), obj_scores.flatten()



def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--image", help="path to image", default='./data/test_data/lena.png')
	parser.add_argument("--conf_thresh", help="confidence of boxes", type=float, default=0.7)
	parser.add_argument("--nms_thresh", help="perform NMS with this threshold", type=float, default=0.5)
	parser.add_argument("--model", help="path to .pt file", default='./logs/yolov3.pt')

	args = parser.parse_args()

	model = load_model(args.model, 1) # 1 for inference

	# Settings
	anchors = [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]
	num_classes = 80 # for pretrained weights
	net_size = (416, 416)

	classes_path = './data/coco.names'
	classes, num_classes = load_classes(classes_path)


	# Preprocessing on image to run inference on
	img, display_image, img_wh = load_image(args.image, net_size)

	loss_obj = Yolov3Loss(anchors, num_classes, net_size)

	all_boxes = []
	all_boxes_scores = []
	all_boxes_conf_scores = []

	with torch.no_grad():
		yolo_out = model(img)
		n_out = len(yolo_out)

		for i in range(len(yolo_out)):
			y_out = loss_obj.processYoloOutput(yolo_out[i], n_out-i-1)
			boxes, scores, conf_scores = get_bounding_boxes(y_out, net_size, img_wh)

			all_boxes.append(boxes)
			all_boxes_scores.append(scores.reshape(-1,80))
			all_boxes_conf_scores.append(conf_scores)


		all_boxes = torch.cat(all_boxes)
		all_boxes_scores = torch.cat(all_boxes_scores)
		all_boxes_conf_scores = torch.cat(all_boxes_conf_scores)
		new_boxes = {}

		bxs = all_boxes[all_boxes_conf_scores>0.7]
		bxs_sc = all_boxes_scores[all_boxes_conf_scores>0.7]
		all_boxes_conf_scores = all_boxes_conf_scores[all_boxes_conf_scores>0.7]

		for i in range(num_classes):
			if(max(bxs_sc[:,i])>args.conf_thresh):
				new_boxes[i] = []
				keep = nms(bxs, bxs_sc[:,i], args.nms_thresh)
				for j in range(len(keep)):
					if(bxs_sc[:,i][keep[j]]>args.conf_thresh):
						new_boxes[i].append((bxs_sc[:,i][keep[j]],bxs[keep[j]]))

	print('Found...')
	print(new_boxes)
	plot_predictions(display_image, new_boxes, classes)

if __name__ == '__main__':
	main()