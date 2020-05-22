import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from model.data_loader import letterbox_image

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import random as rnd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(img_path, net_size, LETTER_BOX=0):

	img = Image.open(img_path)
	img = img.convert('RGB')
	img_w, img_h = img.size

	disp_img = img.copy()

	if LETTER_BOX:
		img = letterbox_image(img, net_size)
	else:
		img = img.resize(net_size)

	transform = transforms.ToTensor()
	img = transform(img) # Change to C,H,W format
	img = img.unsqueeze_(0) # Add batch dimension

	return img.to(device), disp_img, (img_w, img_h)


def load_model(model_path, INFER=1):

	print('Loading pretrained model...')
	model = torch.load(model_path)

	if INFER:
		print('For inference')
		model.to(device).eval()
	else:
		model.to(device)

	return model

def load_classes(fp):

	classes = {}
	count = 0
	with open(fp, 'r') as f:
		for line in f:
			cls = line.rstrip()
			classes[count] = cls
			count+=1

	return classes, count


def plot_predictions(disp_img, boxes, classes, f_name):

	plt.style.use('dark_background')
	fig = plt.figure()
	ax = plt.subplot()
	for pred in boxes.keys():
		R,G,B = rnd.randint(0,255),rnd.randint(0,255),rnd.randint(0,255)
		for b in boxes[pred]:
			score, box = b
			score = round(score.item(),2)
			rect = mpatches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, edgecolor='#%02x%02x%02x'%(R,G,B), linewidth=2.0)
			ax.add_patch(rect)
			ax.annotate(classes[pred]+' '+str(score),(box[0], box[1]-5))
	ax.imshow(disp_img)
	plt.axis('off')
	plt.savefig("./data/test_data/out/"+f_name)
	# plt.show()