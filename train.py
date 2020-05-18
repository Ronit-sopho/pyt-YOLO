import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from model.net import YOLOv3Net, Yolov3Loss
from model.data_loader import FaceDataset, YoloLabel
import gc

def mem_alloc():
	for obj in gc.get_objects():
		try:
			if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
				print(type(obj), obj.size())
		except:
			pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# Read relevant files
annotation_file = './data/annotations.txt'
image_dir = './data/images/originalPics'
# Hard code anchors for now
anchors = [30,43, 60,88, 78,116, 98,148, 124,186, 154,232, 187,280, 222,335, 285,416]
num_anchors = len(anchors)//2
num_classes = 1
input_size = (416,416)
batch_size = 4

# Load Dataset
# Factor out resizing and net size later on!!
transfomations = transforms.Compose([transforms.ToTensor()])
dataset = FaceDataset(annotation_file, image_dir, transform=transfomations, resize=(416,416))
data_holder = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
print('Dataset load complete...')

# Define the model

model = YOLOv3Net(num_classes,num_anchors).to(device)
model.train()
s = 0
for name,p in model.named_parameters():
	if p.requires_grad:
		if 'bn' in name:
			s+=2*p.numel()
		else:
			s+=p.numel()
		print(name, p.shape)
print('Total number of learnable params = ', s)

# Loss function
custom_loss = Yolov3Loss(anchors, num_classes, input_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train cycle
num_steps = len(data_holder)
num_epochs = 100

for epoch in range(num_epochs):
	for i, data in enumerate(data_holder):
		imgs = data['image'].to(device)
		print(imgs.shape)
		lbls = data['label'].to(device)

		# forward pass
		yolo_output = model(imgs)
		# print("yo ", yolo_output[1])
		yolo_loss = custom_loss.loss(yolo_output, YoloLabel(lbls, (416, 416), anchors, num_classes, batch_size))
		# print(torch.cuda.memory_allocated(device=device))

		# backward pass
		optimizer.zero_grad()
		yolo_loss.backward()
		optimizer.step()
		# mem_alloc()
		torch.cuda.empty_cache()
		print("Epoch: {}/{}, Step: {}, Loss: {:.4f}".format(epoch, num_epochs,i+1, yolo_loss.item()))