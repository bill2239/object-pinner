import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from skimage import io, transform
from torch.autograd import Variable
from mymodel import AlexNet
import cv2

def img2tensor(img_name):
	image = io.imread(img_name)
	img = transform.resize(image, (224, 224))
	img = img.transpose((2, 0, 1))
	img = img[None,:,:]
	return img




def output_overlay(img_path,output_arr,label_path):
	label_arr= [line.strip("\n").split(" ") for line in open(label_path,'r').readlines()]
	label_map= {}
	for ele in label_arr:
		label_map[ele[0]]= [float(ele[1]),float(ele[2])]
	image_name=img_path.split("/")[1]
	img=cv2.imread(img_path)
	(h,w,_)= img.shape
	loc_label=(int(label_map[image_name][0]*w),int(label_map[image_name][1]*h))
	loc_predicted= (int(output_arr[0][0]*w),int(output_arr[0][1]*h))
	
	cv2.circle(img,loc_label, 3, (0,0,255), -1)

	cv2.circle(img,loc_predicted, 3, (0,255,0), -1)
	cv2.imwrite("example_overlay.jpg",img)

if __name__ == "__main__" :
	parser = argparse.ArgumentParser()
	parser.add_argument('path')
	args = parser.parse_args()
	net=torch.load('model_289_0.0001_79.pth')
	img_name=args.path
	img=img2tensor(img_name)
	inputs= torch.from_numpy(img).float() 
	inputs= Variable(inputs).cuda()
	outputs= net(inputs)
	output_arr=outputs.data.cpu().numpy()
	#print the x,y location in the image 
	print output_arr[0][0],' ',output_arr[0][1]
	output_overlay(img_name,output_arr,"label.txt")
