# Computer Vision Assignment Report

Title: Computer Vision Assignment 3 Report

Student Name: Zitong Huang

Student ID: 12012710



### 1. Experimental Design

In this experimental, scene classify can be divided into 2 steps: feature building and feature classifing.

#### Feature Building

In this experimental I have used 2 ways to build features: Tiny Image and Sift

##### Tiny Image

​		Tiny Image is introduced by A.Torralba in 2008. It is a simple way to reduce dimension: just resize the image to a lower size, such as 16*16 in the assignment. However, image information will loss a lot by using this way. Thus, we introduced another way to describe an image.

##### SIFT feature description with Word Bag Module

​		Word Bag Module is introduced by Lazebnik in 2006. At first, a M*d array can be attained by SIFT feature description.   However, after that we will get a N * M * d array, which is terrible for classify. To avoid such "Dimension Disaster", Kmeans cluster is used to reduce dimension. At this step, we assign a variable call vocal_size as the number of cluster —— or, the number of "word bags". Then, for each features, we assigned it to a specific word bag. The next step is calculate feature numbers in each bag, and get a t-d vector. Obviously, t in here is equal to vocab_size. with doing so, we can use a t dimension vector to represent an image.

### 2. Experimental Results Analysis

Some result in this experimental is show below:

| vocab_size          | 10   | 20   | 50   | 100  | 200  | 400  | 1000 | 10000 |
| ------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- |
| KNN with Tiny Image |      |      |      |      |      |      |      |       |
| SVM with Tiny Image |      |      |      |      |      |      |      |       |
| KNN with WBM        |      |      |      |      |      |      |      |       |
| SVM with WBM        |      |      |      |      |      |      |      |       |

Other hyper-parameters is show below:

train/test data size: 200

sift size

sift step



### 3. Bonus Report (If you have done any bonus problem, state them here)

