import matplotlib.pyplot as plt
from ipdb import set_trace as debug
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parseData import importData 

joints = ['neck', 'left shoulder','left elbow', 'left wrist','left finger','right shoulder',
		'right elbow','right wrist','right finger','left hip','left knee','left ankle',
		'left toe','right hip','right knee','right ankle','right toe']

n_groups = len(joints)
fig, ax = plt.subplots(figsize=(20, 10))
index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.8

# # clean data
# data = importData('trials/RMSE_mm.txt')
# data_clean = list()
# for i, entry in enumerate(data):
# 	current_row = list()
# 	for j in range(len(entry)):
# 		try:
# 			current_row.append(float(entry[j]))
# 		except:
# 			# if the current entry is not a number then skip it
# 			pass

# 	if len(current_row) > 0:
# 		data_clean.append(current_row)
# rmse = np.array(data_clean[-17:])



# import data
data = importData('trials/rmse_numLayers5_epochs12000_drop0.6_lr0.001.csv')
rmse = np.array(data).reshape((18,3))
rmse = rmse.astype(np.float32)
rmse = rmse[1:,:]


rmse_avg = np.mean(rmse)
print rmse_avg

rects1 = plt.bar(index, rmse[:,0], bar_width,
                 alpha=opacity,
                 color='b',
                 label='X')
 
rects2 = plt.bar(index + bar_width, rmse[:,1], bar_width,
                 alpha=opacity,
                 color='g',
                 label='Y')

rects3 = plt.bar(index + 2*bar_width, rmse[:,2], bar_width,
                 alpha=opacity,
                 color='r',
                 label='Z')
 
plt.xlabel('Joint', fontsize=24)
plt.ylabel('RMSE', fontsize=24)
plt.title('RMSE of NN', fontsize=24)
plt.xticks(index + bar_width, joints, rotation='vertical')
plt.legend(fontsize=24)
 
plt.tight_layout()
plt.savefig('plots/RMSE_NN.png')


