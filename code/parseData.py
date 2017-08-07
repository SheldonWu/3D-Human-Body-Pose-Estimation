'''
Parser to format EgoCap data into Keras VGG-16 Model
    Model Schema is based on 
    https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    ImageNet Pretrained Weights 
    https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of categories for our classification task
'''

from ipdb import set_trace as debug
import numpy as np
import scipy as sp
import random
import csv
import copy
import os
from PIL import Image
import pickle

# create a Matlab like struct class
class DOPEstruct():
    pass

def importData(input_path):
    # import raw data as provided in EgoCap datset
    data_in = []
    with open(input_path) as file:
        for line in file:
            # The rstrip method gets rid of the "\n" at the end of each line
            data_in.append(line.rstrip().split(' '))

    return data_in


def getParams(mode='train', subset_sample_size=20, model='2DTo3D'):
    # define some parameters
    H,W,C = 512,640,3

    params = DOPEstruct()
    params.mode = mode       # 'train' or 'test'
    params.model = model     # 'imgTo2D' or '2DTo3D'
    params.random_subset_size = subset_sample_size
    params.folder_name = 'training_v002' if params.mode == 'train' else 'validation_v003_2D'
    params.body_part_labels = ['head','neck', 'left shoulder', 'left elbow', 'left wrist', 'left finger', 'right shoulder', 'right elbow',
                               'right wrist', 'right finger', 'left hip', 'left knee', 'left ankle', 'left toe', 'right hip', 
                               'right knee', 'right ankle', 'right toe']
    params.num_body_parts = len(params.body_part_labels)
    params.input_labels_path = 'data_original/' + params.folder_name + '/dataset.txt' 
    params.input_image_path = 'data_original/' + params.folder_name    
    params.num_lines_per_example = 24
    params.num_examples = sum(1 for line in open(params.input_labels_path))/params.num_lines_per_example
    params.img_size = H,W,C

    params.input_3d_labels_path = 'data_original/validation_v003_3D/dataset3D.txt'
    params.input_2d_data_path = 'data_original/validation_v003_2D/dataset.txt'


    return params


def load_data3D():
    # load the 3d validation data. Use the 3D labels combined with the 2D labels to generate a 2D to 3D 
    # human body pose estimator

	try: 
		# load data from saved file
		data3D = np.load('data_formatted/data3D.npy')

	except:
		# parse data from orignal file
		params = getParams()

		# read in text file of 3D labels
		data_in = importData(params.input_3d_labels_path)

		num_examples = len(data_in)
		data3D = dict()

		for i in range(num_examples):

			# remove any spaces in the list for each example
			example_data = [x for x in data_in[i] if x !='']

			# check to make sure example i is formatted correctly. If not, skip that example
			if len(example_data) == 55:

				# get the ID number of the ucrrent example
				current_ID = int(example_data[0])  

				# initialize the dictionary as a numpy array for each key (num_joints, 3) for every joint x,y,z coordinate
				data3D[current_ID] =  np.zeros((params.num_body_parts, 3), dtype=np.float32)

				for j in range(params.num_body_parts):     
					# collect the 3D data 
					data3D[current_ID][j,0] = float(example_data[3*j+1])
					data3D[current_ID][j,1] = float(example_data[3*j+2])
					data3D[current_ID][j,2] = float(example_data[3*j+3])
					# print i,j

		# save the data for easy access next time			
		np.save('data_formatted/data3D',data3D)

	return data3D




def load_data2D(mode='train', subset_sample_size = 20, model='2DTo3D', sample=True):


	try: 
		# load data from saved file
		if model == '2DTo3D':
			out = np.load('data_formatted/data2D.npy')
		elif model == 'imgTo2D':
			out = np.load('data_formatted/imgTo2D.npy')
		

	except:
		# load data from original data file then save as formatted for convenient access later
		if model == '2DTo3D':
			mode = 'test'

		# get those parameters daddy!
		params = getParams(mode=mode, subset_sample_size = subset_sample_size, model=model)
		H,W,C = params.img_size

		if model == '2DTo3D':
			# import data from text file
			data_in = importData(params.input_2d_data_path)
		else:
			data_in = importData(params.input_labels_path)

		# initialize variables and dictionaries
		dataset, X, y, labels2D = dict(), dict(), dict(), dict()

	    # random subsample
		if sample:
			sample_indices = random.sample(range(params.num_examples), params.random_subset_size)
		else:
			sample_indices = range(params.num_examples)

		# each body part will have its own classifier so each body part needs separate labels
		for i in range(params.num_body_parts):

			# text for the current body part
			body_part = params.body_part_labels[i]

			# create a struct for each body part all contained within a single dictionary
			dataset[body_part] = DOPEstruct()
			dataset[body_part].pixel_labels = np.zeros((params.num_examples,2), dtype=np.int64)
			dataset[body_part].img_size = np.zeros((params.num_examples,3), dtype=np.int64)
			dataset[body_part].num_joints = np.zeros((params.num_examples,1), dtype=np.int64)
			dataset[body_part].example_id = np.zeros((params.num_examples,1), dtype=np.int64)
			dataset[body_part].labels = np.zeros((params.num_examples,1), dtype=np.int64)
			dataset[body_part].example_name = list()

			# create data and label arrays for each body part
			X[body_part] = np.zeros((params.random_subset_size, C, H, W), dtype=np.uint8)
			y[body_part] = np.zeros((params.random_subset_size, 2*params.num_body_parts), dtype=np.float32)

			# iterate through all the examples
			for nu, j in enumerate(sample_indices):

				# extract current example's ID number
				current_ID = j*params.num_lines_per_example
				dataset[body_part].example_id[j] = int(data_in[current_ID][1]) 


				# extract current example's name
				current_name_row = j*params.num_lines_per_example + 1
				dataset[body_part].example_name.append(data_in[current_name_row][0])

				# extract current example's image size
				current_size_rows = np.arange(2,5) + j*params.num_lines_per_example
				dataset[body_part].img_size[j,:] = (int(data_in[current_size_rows[1]][0]),
				                                    int(data_in[current_size_rows[2]][0]),
				                                    int(data_in[current_size_rows[0]][0]))

				# extract current example's number of joint/body parts (just to make sure all the examples have 18)
				current_num_joints = j*params.num_lines_per_example + 5
				dataset[body_part].num_joints[j] = int(data_in[current_num_joints][0]) 


				# extract current example's pixel labels for a particular body part (x,y)
				current_label_row = j*params.num_lines_per_example + 6 + i
				xy_indices = [2,1] if i == 0 else [1,2] # handle the case where the head coordinates are labeled (y,x) and the rest of the joint coords are (x,y)
				dataset[body_part].pixel_labels[j,:] = (int(data_in[current_label_row][xy_indices[0]]), int(data_in[current_label_row][xy_indices[1]]))

				# formatData4Keras(body_part, dataset[body_part].example_name[-1][1:], dataset[body_part].labels[j], params)
				input_image_path = params.input_image_path + dataset[body_part].example_name[-1][1:]
				img = np.asarray(Image.open(input_image_path))
				img = img.transpose((2,0,1))


				if params.model == 'imgTo2D':
					# Format input data (X) and label (y) for fine-tuning models from https://github.com/fchollet/deep-learning-models/blob/master/
					X[body_part][nu] = img  # rearrange to match import format of opencv
					y[body_part][nu] = dataset[body_part].pixel_labels[j,:]

					# define output
					out = X,y

				elif params.model == '2DTo3D':
					# Format input data for 2D to 3D learning model
					if i == 0:
						labels2D[dataset[body_part].example_id[j][0]] = np.zeros((params.num_body_parts, 2), dtype=np.float32)

					labels2D[dataset[body_part].example_id[j][0]][i,:] = dataset[body_part].pixel_labels[j,:]

					# define output
					out = labels2D

				print i,j

		if params.model == '2DTo3D':
			np.save('data_formatted/data2D',out)
		elif params.model == 'imgTo2D':
			np.save('data_formatted/imgTo2D',out)


	return out


def splitData(data, train_percent=.7):
	data_2D, labels_3D = data
	size_in = np.prod(data_2D[()][0].shape)
	size_out = np.prod(labels_3D[()][4096].shape)
	EgoCap = DOPEstruct()

	# split key indices into train and test
	keys = labels_3D[()].keys()
	all_keys_indx = range(len(keys))
	train_keys_indx = random.sample(all_keys_indx, int(round(train_percent*len(keys))))
	test_keys_indx = list(set(all_keys_indx) - set(train_keys_indx))

	# get actual train and test keys from the indices
	train_keys = [keys[x] for x in train_keys_indx]
	test_keys = [keys[x] for x in test_keys_indx]

	# get actual data and format into its final form
	EgoCap.X_train = np.vstack([data_2D[()][x] for x in train_keys]).reshape((len(train_keys_indx),size_in))

	EgoCap.X_test = np.vstack([data_2D[()][x] for x in test_keys]).reshape((len(test_keys_indx),size_in))

	EgoCap.y_train = np.vstack([labels_3D[()][x] for x in train_keys]).reshape((len(train_keys_indx),size_out))

	EgoCap.y_test = np.vstack([labels_3D[()][x] for x in test_keys]).reshape((len(test_keys_indx),size_out))

	return EgoCap

def getOneExample(example_ID):
	allData2D = np.load('data_formatted/data2D.npy')
	allData3D = np.load('data_formatted/data3D.npy')

	example = allData2D[()][example_ID]
	D = np.prod(example.shape)

	out = np.reshape(example,(1,D)) 

	return out


def getEgoCapData():

	try:
		file_Ego = open('data_formatted/EgoCap_data.obj', 'r')
		EgoCap = pickle.load(file_Ego)

	except:
		# collect data and format
		data_2D = load_data2D(model='2DTo3D', sample=False)
		labels_3D = load_data3D()

		data = data_2D, labels_3D
		EgoCap = splitData(data, train_percent=.7)

		# save data for next time	
		# np.save('EgoCap_data',EgoCap)
		file_Ego = open('data_formatted/EgoCap_data.obj', 'w')
		pickle.dump(EgoCap, file_Ego)

	return EgoCap



if __name__ == "__main__":
	EgoCap = getEgoCapData()


