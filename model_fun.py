import os
from skimage import io
import cv2
import numpy as np
from skimage.measure import label, regionprops, regionprops_table
import math

import matplotlib.pyplot as plt
from skimage import io, measure, color
from skimage import morphology
from skimage import io

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
import warnings
import csv
#predict rectangle shape:
import math
from scipy.optimize import minimize
from scipy.optimize import rosen, differential_evolution
from scipy.spatial.distance import cdist
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from PIL import Image
from PIL import ImageTk as itk

warnings.filterwarnings("ignore")

def resize_image(image, target_size):
    # Resize the image to target size using cv2.resize
    image = cv2.resize(image, (target_size[1], target_size[0]),
                       interpolation=cv2.INTER_NEAREST)
    return image

def image_folder(file_path,target_size):
    target_size = (target_size[0], target_size[1]) # Define the target size
    images = []
    for filename in sorted(os.listdir(file_path)):
        resized_image = resize_image(io.imread(os.path.join(file_path, filename)), target_size)
        images.append(resized_image/255.0)
    images=np.array(images)
    return images


#images=image_folder('/kaggle/input/dataset/images/images');
#plt.imshow(masks[0,:,:,:])


def get_cetroids(segmented_image):
    threshold_image=segmented_image[0,:,:,0]>0.5;
    selem = morphology.square(3)
    threshold_image = morphology.remove_small_objects(threshold_image, min_size=400,in_place=False)
    #fig, ax = plt.subplots()
    #ax.imshow(threshold_image, cmap=plt.cm.gray)
    labels = measure.label(threshold_image)
    regions = measure.regionprops(labels)
    centroids = [region.centroid for region in regions]
    centers_old=np.array(centroids);
    
    aspect_ratios = [region.major_axis_length / region.minor_axis_length for region in regions]
    image=threshold_image;

    #plt.imshow(color.label2rgb(labels, image=threshold_image))
    #for centroid in centroids:
    #    plt.scatter(centroid[1], centroid[0], color='red')
    #for aspect_ratio in aspect_ratios:
    #    plt.text(centroid[0], centroid[1], f'Aspect ratio: {aspect_ratio:.2f}', color='blue', fontsize=8)
    #plt.show()

    mask = np.ones(image.shape, dtype=bool)
    for region, aspect_ratio in zip(regions, aspect_ratios):
        if aspect_ratio < 0.85 or aspect_ratio > 1.3:
            mask[tuple(region.coords.T)] = False

#     plt.figure(figsize=(10, 5))
#     plt.subplot(121)
#     plt.imshow(image, cmap='gray')
#     plt.title('Original')
#     plt.axis('off')

    
#     plt.subplot(122)
#     plt.imshow(filtered_image, cmap='gray')
#     plt.title('Filtered')
#     plt.axis('off')

#     plt.show()       

    ##relabel and reassess centroids:
    filtered_image=image * mask.astype(int);
    labels = measure.label(filtered_image)
    regions = measure.regionprops(labels)
    centroids = [region.centroid for region in regions]

    ###now collect all the centers:
    centers=np.array(centroids);


    return filtered_image, centers;
    


##create grid



# with open("/kaggle/working/centers_"+name_folders[0:-4]+".csv", 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         points.append([float(row[0]), float(row[1])])
        

#name_folders=list_folders[0]
#points = resize_image(io.imread("/kaggle/working/centers_"+name_folders[0:-4]+"_output.png"), target_size)
#points = [[0,0], [1,1], [2,2], [3,3], [4,2], [5,1], [6,2], [7,3]]
def rec_estimate(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]

    ### remove outliers
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x)
    std_y = np.std(y)

    filtered_points = [[x[i], y[i]] for i in range(len(x)) if abs(x[i] - mean_x) < 2 * std_x and abs(y[i] - mean_y) < 2 * std_y]
    x = [p[1] for p in filtered_points]
    y = [p[0] for p in filtered_points]

    coefs = np.polyfit(x, y, 1)
    slope = coefs[0]
    y_intercepts = np.linspace(min(y), max(y), 12)

    for y_intercept in y_intercepts:
        polynomial = np.poly1d([slope, y_intercept])
        ys = polynomial(x)
        #plt.plot(x, ys, '-')

    #plt.plot(x, y, 'o')
    #plt.show()
    ###estimate angle
    angle_est=-math.atan(slope);
    ###estimate a1
    min_index = np.argmin((np.array(x)**2+np.array(y)**2))
    indx_est=x[min_index];
    indy_est=y[min_index];
    
    return angle_est, indx_est, indy_est




def slope_and_intercept(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    y_intercept = y1 - slope * x1
    return slope, y_intercept

def plot_lines(rows, columns, p1_rho, Lx, Ly,dist_circ, indx, indy):
#rows=12, columns=8
#indx
    if abs(p1_rho)<1e-8:
        p1_rho=1e-6;
        
    centersx=np.zeros([rows, columns])
    centersy=np.zeros([rows, columns])
    linesy_left=np.zeros([rows,2]); linesy_right=np.zeros([rows,2]);
    linesx_upper=np.zeros([columns,2]); linesx_lower=np.zeros([columns,2]);

    linesy_left[:,0]=np.linspace(-indy*math.sin(p1_rho),-indy*math.sin(p1_rho)-(rows-1)*dist_circ*math.sin(p1_rho),rows);
    linesy_left[:,1]=np.linspace(indy*math.cos(p1_rho),indy*math.cos(p1_rho)+(rows-1)*dist_circ*math.cos(p1_rho),rows)

    linesx_upper[:,0]=np.linspace(indx*math.cos(p1_rho),indx*math.cos(p1_rho)+(columns-1)*dist_circ*math.cos(p1_rho),columns); 
    linesx_upper[:,1]=np.linspace(indx*math.sin(p1_rho),indx*math.sin(p1_rho)+(columns-1)*dist_circ*math.sin(p1_rho),columns);

    linesy_right[:,0]=linesy_left[:,0]+np.ones([1,rows])*Lx*math.cos(p1_rho);
    linesy_right[:,1]=linesy_left[:,1]+np.ones([1,rows])*Lx*math.sin(p1_rho);

    linesx_lower[:,0]=linesx_upper[:,0]-np.ones([1,columns])*Ly*math.sin(p1_rho);
    linesx_lower[:,1]=linesx_upper[:,1]+np.ones([1,columns])*Ly*math.cos(p1_rho);
    
    linesy_slope=(linesy_right[:,1]-linesy_left[:,1])/(linesy_right[:,0]-linesy_left[:,0]);
    line_coeffx=np.zeros([columns,2]); line_coeffy=np.zeros([rows,2]);
    for i4 in range(columns):
        slope, yint=slope_and_intercept(linesx_upper[i4,0], linesx_upper[i4,1], linesx_lower[i4,0], linesx_lower[i4,1])
        line_coeffx[i4,0]=yint
        line_coeffx[i4,1]=slope
        #[np.ones([1,1]) [linesx_upper([0,i4]);linesx_lower([0,i4])]]\[linesx_upper([1,i4]);linesx_lower([1,i4])];
   
    for j4 in range(rows):
        #line_coeffy[j4,0]=[np.ones([1,1]) [linesy_left([0,j4]);linesy_right([0,j4])]]\[linesy_left([1,j4]);linesy_right([1,j4])];
        slope, yint=slope_and_intercept(linesy_left[j4,0],linesy_left[j4,1], linesy_right[j4,0], linesy_right[j4,1])
        line_coeffy[j4,0]=yint
        line_coeffy[j4,1]=slope
        
    for i4 in range(columns):
    
        for j4 in range(rows):
            b2=line_coeffx[i4,0]; b1=line_coeffy[j4,0];
            m2=line_coeffx[i4,1]; m1=line_coeffy[j4,1];
            centersx[j4,i4]=(b2-b1)/(m1-m2);
            centersy[j4,i4]=m1*centersx[j4,i4]+b1;

    return centersx, centersy


def create_circles(x, y, R, image_shape, color='white'):
    image = np.zeros(image_shape, dtype=np.uint8)
    y_coord, x_coord = np.ogrid[:image_shape[0], :image_shape[1]]
    circle_mask = np.zeros(image_shape, dtype=np.uint8)
    for i in range(x.size):
        mask = (x_coord - x[i])**2 + (y_coord - y[i])**2 <= R**2
        circle_mask[mask] = 1
        image = np.maximum(image, circle_mask)
    #plt.imshow(image, cmap='gray')
    #plt.axis('off')
    #plt.show()
    return image;

#plot_lines(rows, columns, p1_rho, Lx, Ly,dist_circ, indx, indy)


## superimpose mask to segmented area from CNN

def superimpose(masked_test_area,segmented_image):
    difference = np.abs(masked_test_area - segmented_image)                                                
    return difference


def calc_diff(params):
    rows=12; columns=8; R=16; image_shape=(672, 448)
    p1_rho, dist_circ, indx, indy=params
    points=plot_lines(rows, columns, p1_rho, Lx, Ly,dist_circ, indx, indy)
    x=np.array(points[0]); y=np.array(points[1])
    masked_test_area=create_circles(x.ravel(), y.ravel(), R, image_shape, color='white');
    dif=superimpose(masked_test_area,output)
    return dif.sum()



def calc_diff_plot(params):
    rows=12; columns=8; R=16; image_shape=(672, 448)
    p1_rho, dist_circ, indx, indy=params
    points=plot_lines(rows, columns, p1_rho, Lx, Ly,dist_circ, indx, indy)
    x=np.array(points[0]); y=np.array(points[1])
    masked_test_area=create_circles(x.ravel(), y.ravel(), R, image_shape, color='white');
    dif=superimpose(masked_test_area,output)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(131)
    plt.imshow(output, cmap='gray')
    plt.title('output')
    plt.axis('off')

    
    plt.subplot(132)
    plt.imshow(masked_test_area, cmap='gray')
    plt.title('grid mask')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(dif, cmap='gray')
    plt.title('superimposed')
    plt.axis('off')
    #plt.imshow(dif)
    #return dif.sum()


### generate 
def calc_diff_plot2(params):
    rows=12; columns=8; R=16; image_shape=(672, 448)
    p1_rho, dist_circ, indx, indy, Lx, Ly=params
    points=plot_lines(rows, columns, p1_rho, Lx, Ly,dist_circ, indx, indy)
    x=np.array(points[0]); y=np.array(points[1])
    masked_test_area=create_circles(x.ravel(), y.ravel(), R, image_shape, color='white');
    dif=superimpose(masked_test_area,output)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(131)
    plt.imshow(output, cmap='gray')
    plt.title('output')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(masked_test_area, cmap='gray')
    plt.title('grid mask')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(dif, cmap='gray')
    plt.title('superimposed')
    plt.axis('off')
    #plt.imshow(dif)
    #return dif.sum()


def matchingbydist(x,y,x_grid,y_grid):
    raw_data = np.column_stack((x, y))
    grid_data = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    distances = cdist(raw_data, grid_data)
    reordered_raw_indices = np.argmin(distances, axis=1)
    #reordered_raw = raw_data[reordered_raw_indices, :]
    total_distance = np.sum(np.min(distances**2, axis=1))
    return total_distance
    #return reordered_raw



def calc_diff_matching_centroids(params, rows, columns, R, x, y):
    #rows=12; columns=8; R=16; 
    image_shape=(672, 448)
    p1_rho, dist_circ, indx, indy, Lx, Ly=params
    points=plot_lines(rows, columns, p1_rho, Lx, Ly,dist_circ, indx, indy)
    x_grid=np.array(points[0]); y_grid=np.array(points[1])
    dif=matchingbydist(x,y,x_grid,y_grid)
    return dif




def crop_image(image, x1, y1, L):
    # Calculate the left, upper, right, and lower pixel coordinates
    left = int(x1 - L / 2)
    upper = int(y1 - L / 2)
    right = int(x1 + L / 2)
    lower = int(y1 + L / 2)

    # Crop the image
    cropped_image = image[upper:lower, left:right]

    return cropped_image

def normalize_saturation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]; 
    mean_v = np.mean(v)
    
    v = (v / mean_v * 166.4895);
    #v/=np.amax(v)
    #mean_v = np.mean(v)
    #print(mean_v)
    hsv[:,:,2] = (v).astype(np.uint8)
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def normalize_saturation_dw(img):
    #print(img.max())
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    mean_v = np.mean(v)
    v = v / mean_v*0.6529;
    v = np.clip(v, 0, 255).astype(np.uint8)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def convert_size(rows, columns, p1_rho,indx, indy, dist_circ, Lx, Ly, image_shape,dw_img_size, file_path):
    image=io.imread(file_path)
    real_shape=(len(image),len(image[0]));
    coeff=(real_shape[0]/image_shape[0]+ real_shape[1]/image_shape[1])/2
    new_centersx, new_centersy=plot_lines(rows, columns, p1_rho, Lx, Ly,dist_circ, indx, indy)
    new_centersx=new_centersx*coeff;
    new_centersy=new_centersy*coeff;
    dw_images=[];
    
    for ix, jx in zip(new_centersx.ravel(), new_centersy.ravel()):
        image
        dw_images.append(crop_image(image, ix, jx, dw_img_size))
    return dw_images
    
def model_test(image, model,get_centroids):
    segmented_image = model.predict(image) #this must be numpy
    if get_centroids:
        output, centers=get_cetroids(segmented_image);
        return output, centers
    else:
        return segmented_image;

import tensorflow as tf

def model_test_dw(image, model):
    segmented_image = model.predict(image) #this must be numpy
    return segmented_image;


def superimpose6(masked_test_area,segmented_image):
    
    
    #resize:
    #masked_test_area_rsz=cv2.resize(masked_test_area, (42, 28),
                      # interpolation=cv2.INTER_NEAREST)
    #segmented_image_rsz=cv2.resize(segmented_image_rsz, (42, 28),
                      # interpolation=cv2.INTER_NEAREST)
    #calculate difference
    difference = np.abs(masked_test_area - segmented_image)                                                
    return difference


def calc_diff6(params,segmented_image):
    rows=3; columns=2; R=88; image_shape=(672, 448)
    p1_rho, dist_circ, indx, indy, Lx, Ly=params
    points=plot_lines(rows, columns, p1_rho, Lx, Ly,dist_circ, indx, indy)
    x=np.array(points[0]); y=np.array(points[1])
    masked_test_area=create_circles(x.ravel(), y.ravel(), R, image_shape, color='white');
    #plt.imshow(masked_test_area)
    dif=superimpose6(masked_test_area.ravel(),segmented_image)
    return dif.sum()

def calc_diff24(params,segmented_image):
    rows=6; columns=4; R=30; image_shape=(672, 448)
    p1_rho, dist_circ, indx, indy, Lx, Ly=params
    points=plot_lines(rows, columns, p1_rho, Lx, Ly,dist_circ, indx, indy)
    x=np.array(points[0]); y=np.array(points[1])
    masked_test_area=create_circles(x.ravel(), y.ravel(), R, image_shape, color='white');
    #plt.imshow(masked_test_area)
    dif=superimpose6(masked_test_area.ravel(),segmented_image)
    return dif.sum()

def list_wells(plnum):
  
  if plnum==96:
   # create a list of letters from 'a' to 'h'
   letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

   # create a list of numbers from '01' to '12'
   numbers = ['{:02d}'.format(i) for i in range(1, 13)]

   # create an empty list to store the combinations
   combinations = []

   # iterate over the letters and numbers, and combine them in the desired sequence
   for num in numbers:
      for letter in letters:
          combinations.append(letter + num)

  if plnum==6:
   # create a list of letters from 'a' to 'h'
   letters = ['a', 'b']

   # create a list of numbers from '01' to '03'
   numbers = ['{:02d}'.format(i) for i in range(1, 4)]

   # create an empty list to store the combinations
   combinations = []

   # iterate over the letters and numbers, and combine them in the desired sequence
   for num in numbers:
      for letter in letters:
          combinations.append(letter + num)
  
  if plnum==24:
   # create a list of letters from 'a' to 'h'
   letters = ['a', 'b', 'c', 'd']

   # create a list of numbers from '01' to '06'
   numbers = ['{:02d}'.format(i) for i in range(1, 7)]

   # create an empty list to store the combinations
   combinations = []

   # iterate over the letters and numbers, and combine them in the desired sequence
   for num in numbers:
      for letter in letters:
          combinations.append(letter + num)

  combinations.reverse();
  return combinations


def image_splitter(filez,file_path):
   print(file_path)
   image=io.imread(filez+'\\'+file_path);
   i,j,k=image.shape;
   image_left=image[:,0:int(j/2),:] 
   image_right=image[:,int(j/2):int(j),:] 
   img_dir=file_path[0:-4];
   if not os.path.isdir(filez+'\\data\\'+img_dir):
    	os.makedirs(filez+'\\data\\'+img_dir);
   plt.imsave(f"{filez}\\data\\{img_dir}\\{img_dir}_plate1.jpg", image_left)
   plt.imsave(f"{filez}\\data\\{img_dir}\\{img_dir}_plate2.jpg", image_right)


def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None): # raster to rle coding 
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return '' ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return '' ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)): # rle coding to raster
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def bwareaopen(img, min_size, connectivity=8):
        """Remove small objects from binary image (approximation of 
        bwareaopen in Matlab for 2D images).
    
        Args:
            img: a binary image (dtype=uint8) to remove small objects from
            min_size: minimum size (in pixels) for an object to remain in the image
            connectivity: Pixel connectivity; either 4 (connected via edges) or 8 (connected via edges and corners).
    
        Returns:
            the binary image with small objects removed
        """
    
        # Find all connected components (called here "labels")
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            img, connectivity=connectivity)
        
        # check size of all connected components (area in pixels)
        for i in range(num_labels):
            label_size = stats[i, cv2.CC_STAT_AREA]
            
            # remove connected components smaller than min_size
            if label_size < min_size:
                img[labels == i] = 0
                
        return img

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

#target_size = (672, 448)
#model = keras.models.load_model("cp.ckpt")
#print("successfully uploaded plate model")

#image_path="image20220213-162836-Plate1_Wells.png";
#image=io.imread(image_path);
#images=[];
#resized_image = resize_image(image,target_size)
#images.append(resized_image)
#images.append(resized_image)
#images = np.array(images);
#print("successfully uploaded image")
#print(images.shape)
#output, centers=model_test(images[0:1,:,:,:],model)
#print("successfully masked the image")



	