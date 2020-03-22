import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from model.net import YOLOv3Net, Yolov3Loss
from model.data_loader import FaceDataset, YoloLabel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# Read relevant files
annotation_file = './data/annotations.txt'
image_dir = './data/images/originalPics'
# Hard code anchors for now
anchors = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
num_anchors = len(anchors)//2
num_classes = 1
input_size = (416,416)

# Load Dataset
# Factor out resizing and net size later on!!
transfomations = transforms.Compose([transforms.ToTensor()])
dataset = FaceDataset(annotation_file, image_dir, transform=transfomations, resize=(416,416))
data_holder = DataLoader(dataset=dataset, batch_size=2, shuffle=True)
print('Dataset load complete...')

# Define the model
model = YOLOv3Net(num_classes,num_anchors).to(device)
model.train()

# Loss function
custom_loss = Yolov3Loss(anchors, num_classes, input_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train cycle
num_steps = len(data_holder)
num_epochs = 10

for epoch in range(num_epochs):
	for i, data in enumerate(data_holder):
		imgs = data['image'].to(device)
		lbls = data['label'].to(device)

		# forward pass
		yolo_output = model(imgs)
		# print("yo ", yolo_output[1])
		yolo_loss = custom_loss.loss(yolo_output, YoloLabel(lbls, (416, 416), anchors, 1))

		# backward pass
		optimizer.zero_grad()
		yolo_loss.backward()
		optimizer.step()
		print(yolo_loss)