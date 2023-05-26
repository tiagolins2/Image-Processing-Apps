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
import csv
from scipy.spatial import distance
from skimage.measure import label, regionprops

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


def import_model_identify_plate(image_path,model_plateid):
    tf.get_logger().setLevel('ERROR')
    target_size = (32*12, 32*8)
    #model_plateid = keras.models.load_model("cp_plate_id.ckpt",compile=False)
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
        pid=0
    elif max_index == 1:
        pid=6
    elif max_index == 2:
        pid=24
    else: 
        pid=96;
    print(f"Identified {pid}-Well Plate")
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


def import_model(image_path,model,plnum,features):
    tf.get_logger().setLevel('ERROR')
    target_size = (672, 448)
    #model = keras.models.load_model(model_path,compile=False)
    
    
     
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
    #print("successfully uploaded image")
    print(f"Image shape = {images.shape}")
    
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
    if plnum==24:
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
    if plnum==24:
      angle_est=p1_rho; indx_est=indx; indy_est=indy;
      params=([p1_rho, dist_circ, indx, indy])
      bounds=[(angle_est-0.03,angle_est+0.03), (dist_circ-30, dist_circ+20), (indx_est-15, indx_est+15),(indy_est-15, indy_est+15),(Lx-15, Lx+15), (Ly-15, Ly+15)]
      result2 =differential_evolution(calc_diff24, bounds, args=(output.ravel(),))

    #calc_diff_plot2(result2.x)
    print(f"Results: {result2}")
    p1_rho, dist_circ, indx, indy, Lx, Ly= result2.x;  
    print(f"p1_rho = {p1_rho}, dist_circ = {dist_circ}, indx = {indx}, indy = {indy}, Lx = {Lx}, Ly = {Ly}");
    points=plot_lines(rows, columns, p1_rho, Lx, Ly,dist_circ, indx, indy)
    x=np.array(points[0]); y=np.array(points[1])
    masked_test_area=create_circles(x.ravel(), y.ravel(), R, image_shape, color='white')
    plt.imsave(f"{img_dir}/mask.png", masked_test_area, cmap='gray')
    dw_img_size=dw_size_raw
    dw_images=convert_size(rows, columns, p1_rho, indx, indy, dist_circ, Lx, Ly, target_size, dw_img_size, file_path)
    #print(len(dw_images))
    #print((output).shape)
    

    if plnum==96:
       model_dw = keras.models.load_model("cp96plate_dw.ckpt")
    if plnum==6:
       model_dw = keras.models.load_model("cp6dw.ckpt",compile=False) 
    if plnum==24:
       model_dw = keras.models.load_model("cp24dw.ckpt",compile=False) 

       #m=0; # do nothing for now
    #image_dw2model=[]
    #for idx in range(len(dw_images)):
    #    image_dw2model.append(resize_image(dw_images))
    if plnum==0:
      try:
       # remove the directory using the rmdir() method
        os.rmdir(img_dir)
        print("Previous directory has been deleted successfully!")
      except OSError as error:
        print(f"Error: {img_dir} : {error.strerror}")
    
    #image_dw2model=np.array(image_dw2model)
    area=[]; length=[]; greenness=[]; 
    num=[]; circularity=[]; major_axis_length=[]; minor_axis_length=[]; ferret_number=[]; encoded_masks=[];
    for idx in range(len(dw_images)):
        
      norm_dw_image=normalize_saturation((dw_images[idx]))/255;
      ###commented to reduce memory
      #plt.imsave(f"{img_dir}/{list_wellsv[idx]}.png", dw_images[idx])
      
      #if method_seg==1: 
      if 'model_dw' in locals():
         mask = import_model_dw(image_path,norm_dw_image,model_dw,(dw_size, dw_size))
      else:
         mask = np.ones([1,len(norm_dw_image[0]), len(norm_dw_image[1]),1]);
      threshold_image=(mask[0,:,:,0]>0.5);
      threshold_image=resize_image(threshold_image*1, (dw_size_raw,dw_size_raw)) 
      threshold_image=delete_far_blobs(threshold_image, plnum)
      ###commented to reduce memory
      plt.imsave(f"{img_dir}/{list_wellsv[idx]}_mask.png", threshold_image,cmap='gray')
      plt.imsave(f"{img_dir}/{list_wellsv[idx]}.png", norm_dw_image)
      area.append(cv2.countNonZero(threshold_image))
      greenness.append(relative_greenness(resize_image(norm_dw_image*1.0, (dw_size_raw,dw_size_raw)), threshold_image));
      num_blobs, avg_circularity, avg_major_axis_length, avg_minor_axis_length, avg_ferret_number=morphological_properties(threshold_image)
      encoded_mask = rle_encode(threshold_image); encoded_masks.append(encoded_mask);
      num.append(num_blobs); circularity.append(avg_circularity); major_axis_length.append(avg_major_axis_length); minor_axis_length.append(avg_minor_axis_length); ferret_number.append(avg_ferret_number); 
    combined_list = list(zip(list_wellsv, area, greenness,num,circularity,major_axis_length,minor_axis_length,ferret_number,encoded_masks))
    header = ['well_id', 'area', 'greenness','num_of_particles','circularity','major_axis_length','minor_axis_length','ferret_number','encoded_mask'];
    with open(f"{img_dir}/data.csv", mode='w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(header)
      writer.writerows(combined_list) 
    #np.savetxt(f"{img_dir}/area.csv", np.array((list_wellsv,area,greenness,)), delimiter=",")
    return masked_test_area, dw_images, area

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


def segment_wells(raw_img,h_max,h_min,s_max,s_min,v_max,v_min,gt,remove_small):
       
       norm_dw_image=normalize_saturation((raw_img))
       lower_color = np.array([h_min, s_min, v_min])
       upper_color = np.array([h_max, s_max, v_max])
       hsv_image = cv2.cvtColor((norm_dw_image), cv2.COLOR_RGB2HSV)
       # Apply the color thresholding to the image
       thresholded_image = cv2.inRange(hsv_image, lower_color, upper_color)
       #print(np.max(hsv_image[:,:,0]))
       #print(np.max(hsv_image[:,:,1]))
       #print(np.max(hsv_image[:,:,2]))
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
       #thresholded_image2 = morphology.remove_small_objects(thresholded_image2, min_size=400,in_place=False)
       if remove_small==1:
         thresholded_image2=bwareaopen(np.array(thresholded_image2,dtype=np.uint8), min_size=400)

       min_size = 506
       
        
       #boundaries = measure.label(thresholded_image2, connectivity=2)
       #props = measure.regionprops(boundaries)
       #for prop in props:
       # if prop.area < min_size:
       #  for coord in prop.coords:
       #     boundaries[coord[0], coord[1]] = 0
       
            
       boundaries2 = ndimage.binary_fill_holes(thresholded_image2.astype(bool)).astype(int)
       inv_boundaries_filled_holess=1-boundaries2;
       inv_boundaries_unfilled_holes=1-thresholded_image2;
       holes=inv_boundaries_unfilled_holes-inv_boundaries_filled_holess;
       holes2keep=morphology.remove_small_objects(thresholded_image2, min_size=min_size,in_place=False)
       boundaries2=boundaries2-holes2keep;
            
       blurryImage = boundaries2;

       windowSize = 8
       kernel = np.ones((windowSize, windowSize)) / windowSize ** 2
       blurryImage = convolve2d(thresholded_image2.astype(float), kernel, mode='same')
       binaryImage = (blurryImage > 0.5)*1
       if remove_small==1:
         binaryImage=bwareaopen(np.array(binaryImage,dtype=np.uint8), min_size=400)
       #binaryImage = morphology.remove_small_objects(binaryImage, min_size=400,in_place=False); 
       
       img = image_array;
       
       img[:,:,0] = np.squeeze(image_array[:,:,0])*binaryImage 
       img[:,:,1] = np.squeeze(image_array[:,:,1])*binaryImage 
       img[:,:,2] = np.squeeze(image_array[:,:,2])*binaryImage 
       
       area = cv2.countNonZero(binaryImage)
       
       return img, area
    
def relative_greenness(img, mask):
    green_channel = img[:, :, 1]
    #mask=np.expand_dims(mask, -1)
    # Apply the mask to the green channel
    masked_green = np.squeeze(img[:,:,1])*mask*1.0
    
    # Calculate the number of green pixels in the masked image
    green_pixels = np.sum(masked_green)

    # Calculate the total number of pixels in the masked image
    total_pixels = np.count_nonzero(mask)

    # Calculate the relative greenness of the masked image
    if total_pixels>0:
      relative_greenness = (green_pixels / total_pixels) * 1.0
    else:
      relative_greenness=0;

    return relative_greenness;

def delete_far_blobs(binary_image, plnum):
    # Calculate center of image
    rows, cols = binary_image.shape
    center_row, center_col = rows // 2, cols // 2
    
    if plnum==96:
       D=60;
    elif plnum==6:
       D=450
    elif plnum==24:
       D=120
    
    # Label the connected components (blobs) in the binary image
    labeled_image = label(binary_image)
    
    # Iterate through each labeled blob
    for region in regionprops(labeled_image):
        # Calculate centroid of the blob
        centroid_row, centroid_col = region.centroid
        
        # Calculate distance from centroid to center of image
        dist = distance.euclidean((centroid_row, centroid_col), (center_row, center_col))
        
        # If distance is greater than D, delete the blob from the labeled image
        if dist > D:
            labeled_image[labeled_image == region.label] = 0
    
    # Convert the labeled image back to a binary image
    binary_image = np.zeros_like(binary_image)
    binary_image[labeled_image > 0] = 1
    
    return binary_image


def morphological_properties(binary_image):
    # Calculate labeled regions in the binary image
    labeled_image = measure.label(binary_image)

    # Calculate properties of the labeled regions
    properties = measure.regionprops(labeled_image)

    circularities = []
    major_axis_lengths = []
    minor_axis_lengths = []
    ferret_numbers = []
    if labeled_image.max()>0: 
     # Loop through each labeled region and calculate the requested properties
     for region in properties:
        # Calculate circularity as 4pi(area/perimeter^2)
        circularity = (4 * np.pi * region.area) / (region.perimeter ** 2)
        circularities.append(circularity)

        # Append length of the major and minor axis to the corresponding lists
        major_axis_lengths.append(region.major_axis_length)
        minor_axis_lengths.append(region.minor_axis_length)
        #major_axis_lengths=0; minor_axis_lengths=0; ferret_number=0;

        # Calculate the ferret numbers as the ratio of the maximum and minimum diameters
        ferret_number = region.major_axis_length / (region.minor_axis_length+0.001)
        ferret_numbers.append(ferret_number)


     avg_circularity = np.mean(circularities)
     avg_major_axis_length = np.mean(major_axis_lengths)
     avg_minor_axis_length = np.mean(minor_axis_lengths)
     avg_ferret_number = np.mean(ferret_numbers)
    else:
     avg_circularity = 0
     avg_major_axis_length = 0
     avg_minor_axis_length = 0
     avg_ferret_number = 0
    return labeled_image.max(), avg_circularity, avg_major_axis_length, avg_minor_axis_length, avg_ferret_number

	