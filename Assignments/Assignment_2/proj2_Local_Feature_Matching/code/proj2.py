import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from student_feature_matching import match_features
from student_sift import get_features
from student_harris import get_interest_points
from skimage import transform
from IPython.core.debugger import set_trace

img_name='RISHLibrary'
feature_width = 16 # width and height of each local feature, in pixels. 
feature_num = 3000 # ANMS args
scale_factor = 0.5

def setup_image(img_name):
	'''
	Func:
		Load image
		Resize image
		Convert pictures to grayscale
		
	'''
	image1 = load_image('../data/'+img_name+'/'+img_name+'1.jpg')
	image2 = load_image('../data/'+img_name+'/'+img_name+'2.jpg')
	eval_file = '../data/'+img_name+'/'+img_name+'Eval.mat'
	image1 = cv2.resize(image1, (0, 0), fx=scale_factor, fy=scale_factor)
	image2 = cv2.resize(image2, (0, 0), fx=scale_factor, fy=scale_factor)
	image1_bw = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
	image2_bw = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
	return image1,image2,image1_bw,image2_bw
	




if __name__ == "__main__":
	# set up images
	image1,image2,image1_bw,image2_bw = setup_image(img_name)
	# Find the interest points
	x1, y1 = get_interest_points(image1_bw,img_name,feature_width,feature_num)
	x2, y2 = get_interest_points(image2_bw,img_name,feature_width,feature_num)
	# save the interest points
	c1 = show_interest_points(image1, x1, y1)
	save_image('../results/'+img_name+'1'+'interest_points.jpg', c1)
	c2 = show_interest_points(image2, x2, y2)
	save_image('../results/'+img_name+'2'+'interest_points.jpg', c2)
	# Create feature vectors at each interest point
	image1_features = get_features(image1_bw, x1, y1, feature_width)
	image2_features = get_features(image2_bw, x2, y2, feature_width)
	# Match features
	matches, confidences = match_features(image1_features, image2_features, x1, y1, x2, y2)
	print('{:d} matches from {:d} corners'.format(len(matches), len(x1)))
	num_pts_to_visualize = 100
	c1 = show_correspondence_circles(image1, image2,
						x1[matches[:num_pts_to_visualize, 0]], y1[matches[:num_pts_to_visualize, 0]],
						x2[matches[:num_pts_to_visualize, 1]], y2[matches[:num_pts_to_visualize, 1]])
	save_image('../results/'+img_name+'_sift.jpg', c1)
	c2 = show_correspondence_lines(image1, image2,
						x1[matches[:num_pts_to_visualize, 0]], y1[matches[:num_pts_to_visualize, 0]],
						x2[matches[:num_pts_to_visualize, 1]], y2[matches[:num_pts_to_visualize, 1]])
	save_image('../results/'+img_name+'_match.jpg', c2)
	# Compare to the correct result
	num_pts_to_evaluate = 100
	try:
		_, c = evaluate_correspondence(image1, image2, '../data/'+img_name+'/'+img_name+'Eval.pkl', scale_factor,
					x1[matches[:num_pts_to_evaluate, 0]], y1[matches[:num_pts_to_evaluate, 0]],
					x2[matches[:num_pts_to_evaluate, 1]], y2[matches[:num_pts_to_evaluate, 1]])
		save_image('../results/'+img_name+'_match.jpg', c)
	except:
		pass

