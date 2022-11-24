import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from student_feature_matching import match_features
from student_sift import get_features
from student_harris import get_interest_points
from IPython.core.debugger import set_trace
import matplotlib.image as mpimg
import scipy.io as scio

img_name='Chase'
try:
	with open('../data/'+img_name+'/'+img_name+'Eval.pkl', 'rb') as f:
		print(f)
		d = pickle.load(f, encoding='latin1')
	x1 = d['x1'].squeeze()
	y1 = d['y1'].squeeze()
	x2 = d['x2'].squeeze()
	y2 = d['y2'].squeeze()
except:
	pass