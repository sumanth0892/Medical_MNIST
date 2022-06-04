import os
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision import datasets, transforms, models

class MedicalMNIST:
	def __init__(self, df, root_dir, transform = None):
		self.annotations = df
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, index):
		img_path = Path.home()/self.root_dir/self.annotations.iloc[index, 0]
		y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
		if self.transform:
			image = self.transform(image)
		return (image, y_label)

def get_labels_df(root_dir):
	root_dir = Path.home()/root_dir
	mp = {}; df = []; label = 0
	for category in os.listdir(root_dir):
		if category[0] == '.':
			continue
		mp[category] = label
		images_path = root_dir / category
		for image in os.listdir(images_path):
			df.append([category + '/' + image, mp[category]])
		label += 1
	df = np.array(df)
	df = pd.DataFrame(df)
	print(df.head())
	return df

def data_transform():
	train_transform = transforms.Compose([
		transforms.RandomRotation(10),
		transforms.RandomHorizontalFlip(),
		transforms.Resize(60),
		transforms.CenterCrop(60),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
	return data_transform





