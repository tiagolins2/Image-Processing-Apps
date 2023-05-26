import os
import tensorflow as tf
from tensorflow import keras
from model_fun import*
#os.system('model_fun.py')
from skimage import io
from scipy import ndimage
from skimage import measure
import numpy as np     
from scipy.signal import convolve2d

import matplotlib.pyplot as plt

tf.autograph.set_verbosity(3)

#target_size = (672, 448)
#model = keras.models.load_model("cp.ckpt")
#print("successfully uploaded plate model")

#image_path="image20220213-162836-Plate1_Wells.png";
#image=io.imread(image_path);
#resized_image = resize_image(image)
#images = np.array(resized_image);

#images=image_folder('/kaggle/input/dataset/images/images',target_size);
#masks=image_folder('/kaggle/input/dataset/masks/masks',target_size);
#list_folders=sorted(os.listdir('/kaggle/input/dataset/images/images'))
#masks = np.expand_dims((masks), axis=-1)


# tf.get_logger().setLevel('ERROR')
# target_size = (672, 448)
# model = keras.models.load_model("cp_96plate.ckpt")
# print("successfully uploaded plate model")

# image_path="Plate6_Wells.png";
# file_path=image_path;
# img_dir=image_path[0:-4];
# if not os.path.isdir(img_dir):
# 	os.makedirs(img_dir); 
# image=io.imread(image_path);
# images=[];
# resized_image = resize_image(image,target_size)
# images.append(resized_image/255.0)
# images.append(resized_image/255.0)
# images = np.array(images);
# print("successfully uploaded image")
# print(images.shape)
# output, centers=model_test(images[0:1,:,:,:],model)
# np.savetxt(f"{img_dir}/centers.csv", centers, delimiter=",")
# plt.imsave(f"{img_dir}/output.png", output, cmap='gray')
# print("successfully masked the image")


# ### start finding wells
# rows=12; columns=8; p1_rho=0.00009; Lx=3.20574346e+02; Ly=3.90657484e+02; dist_circ=50; indx=50; indy=50; R=15; image_shape=(672, 448);
# initial_guess = np.array([0.00009, 320, 500, 40, 90, 90]); #Ly=4.30657484e+02
# x=centers[:,1]; y=centers[:,0]
# angle_est, indx_est,indy_est=rec_estimate(centers);
# bounds=[(angle_est-0.03,angle_est+0.03), (30, 50), (indx_est-10, indx_est+10),(indy_est-10, indy_est+10), (Lx-10, Lx+10), (Ly-10, Ly+10)]
# result2 =differential_evolution(calc_diff_matching_centroids, bounds, args=(12, 8, 16, x, y))
# #calc_diff_plot2(result2.x)
# p1_rho, dist_circ, indx, indy, Lx, Ly= result2.x;  
# dw_img_size=195
# dw_images=convert_size(rows, columns, p1_rho, indx, indy, dist_circ, Lx, Ly, target_size, dw_img_size, file_path)
# print(len(dw_images))

# for idx in range(len(dw_images)):
# 	plt.imsave(f"{img_dir}/well_{idx}.png", dw_images[idx])


def import_model_identify_plate(image_path):
    tf.get_logger().setLevel('ERROR')
    target_size = (672, 448)
    model_plateid = keras.models.load_model("cp_plate_id.ckpt")
    file_path=image_path;
    img_dir=image_path[0:-4];
    if not os.path.isdir(img_dir):
    	os.makedirs(img_dir); 
    image=io.imread(image_path);
    images=[];
    resized_image = resize_image(image,target_size)
    images.append(resized_image/255.0)
    images.append(resized_image/255.0)
    images = np.array(images);
    print("successfully uploaded image")
    print(images.shape)
    val=model_plateid.predict(images[0:1,:,:,:])
    max_index = np.argmax(val)

    # Output the corresponding number
    if max_index == 0:
        pid=6
    elif max_index == 1:
        pid=24
    else:
        pid=96
    print(pid)
    return pid

def import_model_dw(image_path,image_dw,model_dw,target_size):
    #target_size = (192, 192)
    
    file_path=image_path;
    image=[];
    image.append(resize_image(image_dw,target_size));
    image.append(resize_image(image_dw,target_size));
    image=np.array(image)
    
    output=model_test_dw(image[0:1,:,:,:],model_dw)
    #img_dir=image_path[0:-4];
    #if not os.path.isdir(img_dir):
    #	os.makedirs(img_dir); 
    return output;
    #plt.imsave(f"{img_dir}/output.png", output, cmap='gray')


def import_model(image_path,model_path,plnum,features):
    tf.get_logger().setLevel('ERROR')
    target_size = (672, 448)
    model = keras.models.load_model(model_path,compile=False)
    print("successfully uploaded plate model")
    
     
    #image_path="Plate6_Wells.png";
    file_path=image_path;
    img_dir=image_path[0:-4];
    if not os.path.isdir(img_dir):
    	os.makedirs(img_dir); 
    image=io.imread(image_path);
    images=[];
    resized_image = resize_image(image,target_size)
    images.append(resized_image/255.0)
    images.append(resized_image/255.0)
    images = np.array(images);
    print("successfully uploaded image")
    print(images.shape)
    
    if plnum==96:
      output, centers=model_test(images[0:1,:,:,:],model,1)
      np.savetxt(f"{img_dir}/centers.csv", centers, delimiter=",")
      plt.imsave(f"{img_dir}/output.png", output, cmap='gray')
      print("successfully masked the image")
      x=centers[:,1]; y=centers[:,0]
      list_wellsv=list_wells(plnum);
    if plnum==6:
      output = model_test(images[0:1,:,:,:],model,0)
      output=np.squeeze(output>0.5);
      print(output.shape)
      plt.imsave(f"{img_dir}/output.png", output, cmap='gray')
      print("successfully masked the image")
      list_wellsv=list_wells(plnum); 
    ### start finding wells
    #rows=12; columns=8; p1_rho=0.00009; Lx=3.20574346e+02; Ly=3.90657484e+02; dist_circ=50; indx=50; indy=50; R=15; 
    rows, columns, p1_rho, Lx, Ly, dist_circ, indx, indy, R,dw_size,dw_size_raw = features 
    image_shape=(672, 448);
    initial_guess = np.array([p1_rho, Lx, Ly, dist_circ, indx, indy]); #Ly=4.30657484e+02
    
    if plnum==96:
      angle_est, indx_est,indy_est=rec_estimate(centers);
      bounds=[(angle_est-0.03,angle_est+0.03), (dist_circ-20, dist_circ+10), (indx_est-10, indx_est+10),(indy_est-10, indy_est+10), (Lx-10, Lx+10), (Ly-10, Ly+10)]
      result2 =differential_evolution(calc_diff_matching_centroids, bounds, args=(rows, columns, R, x, y))
    if plnum==6:
      angle_est=p1_rho; indx_est=indx; indy_est=indy;
      params=([p1_rho, dist_circ, indx, indy])
      bounds=[(angle_est-0.03,angle_est+0.03), (dist_circ-50, dist_circ+30), (indx_est-25, indx_est+25),(indy_est-25, indy_est+25),(Lx-20, Lx+20), (Ly-20, Ly+20)]
      result2 =differential_evolution(calc_diff6, bounds, args=(output.ravel(),))
    #calc_diff_plot2(result2.x)
    print(result2)
    p1_rho, dist_circ, indx, indy, Lx, Ly= result2.x;  
    print(f"p1_rho = {p1_rho}, dist_circ = {dist_circ}, indx = {indx}, indy = {indy}, Lx = {Lx}, Ly = {Ly}");
    points=plot_lines(rows, columns, p1_rho, Lx, Ly,dist_circ, indx, indy)
    x=np.array(points[0]); y=np.array(points[1])
    masked_test_area=create_circles(x.ravel(), y.ravel(), R, image_shape, color='white')
    plt.imsave(f"{img_dir}/mask.png", masked_test_area, cmap='gray')
    dw_img_size=dw_size_raw
    dw_images=convert_size(rows, columns, p1_rho, indx, indy, dist_circ, Lx, Ly, target_size, dw_img_size, file_path)
    print(len(dw_images))
    print((output).shape)
    

    if plnum==96:
       model_dw = keras.models.load_model("cp96plate_dw.ckpt")
    if plnum==6:
       m=0; # do nothing for now
    #image_dw2model=[]
    #for idx in range(len(dw_images)):
    #    image_dw2model.append(resize_image(dw_images))
        
    #image_dw2model=np.array(image_dw2model)
    for idx in range(len(dw_images)):
        
      norm_dw_image=normalize_saturation((dw_images[idx]))/255;
      ###commented to reduce memory
      #plt.imsave(f"{img_dir}/{list_wellsv[idx]}.png", dw_images[idx])
      
      #if method_seg==1: 
      if 'model_dw' in locals():
         mask = import_model_dw(image_path,norm_dw_image,model_dw,(dw_size, dw_size))
      else:
         mask = np.ones([1,len(norm_dw_image[0]), len(norm_dw_image[1]),1]);
      threshold_image=mask[0,:,:,0]>0.5;
      ###commented to reduce memory
      #plt.imsave(f"{img_dir}/{list_wellsv[idx]}_mask.png", threshold_image,cmap='gray')
      area = cv2.countNonZero(mask) 
    return masked_test_area, dw_images

def apply_threshold_dw(h_min,h_max,s_min,s_max,v_min,v_max,GT,img):
       
       # Define the lower and upper color threshold values as numpy arrays
       lower_color = np.array([h_min, s_min, v_min])
       upper_color = np.array([h_max, s_max, v_max])
       hsv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
       # Apply the color thresholding to the image
       thresholded_image = cv2.inRange(hsv_image, lower_color, upper_color)
       image_array=np.array(img);
       red_channel = image_array[:, :, 0]
       green_channel = image_array[:, :, 1]
       blue_channel = image_array[:, :, 2]
       # Calculate the green intensity of each pixel relative to the red and blue values
       green_intensity = (255 + green_channel - ((red_channel)*0.5 + (blue_channel)*0.5))*0.5
       thresholded_image2 = (green_intensity>float(GT))*1
     
       print(thresholded_image2.shape)
       print(type(thresholded_image2))
       print(hsv_image.shape)
       
       thresholded_image2=thresholded_image2+thresholded_image
       thresholded_image2=(thresholded_image2>=1)*1
       img = image_array;
       img[:,:,0] = np.squeeze(image_array[:,:,0])*thresholded_image2
       img[:,:,1] = np.squeeze(image_array[:,:,1])*thresholded_image2
       img[:,:,2] = np.squeeze(image_array[:,:,2])*thresholded_image2
       img = Image.fromarray(img)
       
       return img;
       


def import_model_6(image_path):
    tf.get_logger().setLevel('ERROR')
    target_size = (672, 448)
    model = keras.models.load_model("cp_96plate.ckpt")
    print("successfully uploaded plate model")

    #image_path="Plate6_Wells.png";
    file_path=image_path;
    img_dir=image_path[0:-4];
    if not os.path.isdir(img_dir):
    	os.makedirs(img_dir); 
    image=io.imread(image_path);
    images=[];
    resized_image = resize_image(image,target_size)
    images.append(resized_image/255.0)
    images.append(resized_image/255.0)
    images = np.array(images);
    print("successfully uploaded image")
    print(images.shape)
    output, centers=model_test(images[0:1,:,:,:],model,1)
    np.savetxt(f"{img_dir}/centers.csv", centers, delimiter=",")
    plt.imsave(f"{img_dir}/output.png", output, cmap='gray')
    print("successfully masked the image")

     
    
    ### start finding wells
    rows=3; columns=2; p1_rho=0.00009; Lx=3.20574346e+02; Ly=3.90657484e+02; dist_circ=186; indx=150; indy=150; R=88; image_shape=(672, 448);
    #initial_guess = np.array([0.00009, 320, 500, 40, 90, 90]); #Ly=4.30657484e+02
    x=centers[:,1]; y=centers[:,0]
    angle_est, indx_est,indy_est=rec_estimate(centers);
    bounds=[(angle_est-0.03,angle_est+0.03), (dist_circ-50, dist_circ+50), (indx_est-50, indx_est+50),(indy_est-50, indy_est+50), (Lx-10, Lx+10), (Ly-10, Ly+10)]
    result2 =differential_evolution(calc_diff_matching_centroids, bounds, args=(3, 2, R, x, y))
    #calc_diff_plot2(result2.x)
    p1_rho, dist_circ, indx, indy, Lx, Ly= result2.x;  
    dw_img_size=195
    dw_images=convert_size(rows, columns, p1_rho, indx, indy, dist_circ, Lx, Ly, target_size, dw_img_size, file_path)
    print(len(dw_images))
    print((output).shape)
    model_dw = keras.models.load_model("cp96plate_dw.ckpt")
    #image_dw2model=[]
    #for idx in range(len(dw_images)):
    #    image_dw2model.append(resize_image(dw_images))
        
    #image_dw2model=np.array(image_dw2model)
    for idx in range(len(dw_images)):
        
      norm_dw_image=normalize_saturation((dw_images[idx])*255)/255;
      
      plt.imsave(f"{img_dir}/well_{idx}.png", dw_images[idx])
      
      mask = import_model_dw(image_path,norm_dw_image,model_dw)
      threshold_image=1-mask[0,:,:,0]>0.5;
      #print(mask.shape)
      plt.imsave(f"{img_dir}/well_{idx}_mask.png", threshold_image,cmap='gray')
    return output, dw_images

#import_model("Plate6_Wells.png")


def segment_wells(raw_img,h_max,h_min,s_max,s_min,v_max,v_min,gt):
       
       norm_dw_image=normalize_saturation((raw_img))
       lower_color = np.array([h_min, s_min, v_min])
       upper_color = np.array([h_max, s_max, v_max])
       hsv_image = cv2.cvtColor((norm_dw_image), cv2.COLOR_RGB2HSV)
       # Apply the color thresholding to the image
       thresholded_image = cv2.inRange(hsv_image, lower_color, upper_color)
       image_array=np.array(norm_dw_image);
       red_channel = image_array[:, :, 0]
       green_channel = image_array[:, :, 1]
       blue_channel = image_array[:, :, 2]
       # Calculate the green intensity of each pixel relative to the red and blue values
       green_intensity = (255 + green_channel - ((red_channel)/2 + (blue_channel)/2))/2
       thresholded_image2 = (green_intensity>float(gt))*1

       thresholded_image2=thresholded_image2+thresholded_image
       thresholded_image11=thresholded_image2;
       thresholded_image2=(thresholded_image2>=1)*1
       
       # apply shape dependent filter
       selem = morphology.square(3)
       thresholded_image2 = morphology.remove_small_objects(thresholded_image2, min_size=400,in_place=False)
       
       min_size = 506
       boundaries = measure.label(thresholded_image2, connectivity=2)
       props = measure.regionprops(boundaries)
       for prop in props:
        if prop.area < min_size:
         for coord in prop.coords:
            boundaries[coord[0], coord[1]] = 0
            
       boundaries2 = ndimage.binary_fill_holes(boundaries.astype(bool)).astype(int)
       bholes = boundaries2 - boundaries
       bholes2 = measure.label(bholes, connectivity=2)
       props = measure.regionprops(bholes2)
       for prop in props:
        if prop.area < min_size:
         for coord in prop.coords:
            bholes2[coord[0], coord[1]] = 1
            
       boundaries = boundaries2 - bholes2       
       thresholded_image2 = boundaries

       windowSize = 8
       kernel = np.ones((windowSize, windowSize)) / windowSize ** 2
       blurryImage = convolve2d(thresholded_image2.astype(float), kernel, mode='same')
       binaryImage = blurryImage > 0.5
       
       img = image_array;
       
       img[:,:,0] = np.squeeze(image_array[:,:,0])*binaryImage 
       img[:,:,1] = np.squeeze(image_array[:,:,1])*binaryImage 
       img[:,:,2] = np.squeeze(image_array[:,:,2])*binaryImage 
       
       area = cv2.countNonZero(thresholded_image2)
       
       return thresholded_image2, area
    
    

	