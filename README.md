# Image-Processing-Apps

# 1. DuckPlate Python app
First open the application, either the standalone app (.exe) or the python app:

<img width="500" alt="tutorial_1_pro" src="https://github.com/tiagolins2/Image-Processing-Apps/assets/95873122/c5c17601-0cda-42ba-8819-d89b6ee8522d">

You can either choose one file or multiple. The files must be sized 4056x3040, with up to 2 well plates

<img width="608" alt="tutorial_2_pro" src="https://github.com/tiagolins2/Image-Processing-Apps/assets/95873122/d66481a0-fd4a-4d86-8e52-4b8dd0338f06">

Once you click Open, the main application will appear on the screen

<img width="600" alt="tutorial_3_pro" src="https://github.com/tiagolins2/Image-Processing-Apps/assets/95873122/6a8466b8-97bb-43f4-b0d6-095272e2f328">

Press Run. You will be able to see the progress of the image processing. Once completed, you can view the results by clicking the button Check results

<img width="600" alt="tutorial_4_pro" src="https://github.com/tiagolins2/Image-Processing-Apps/assets/95873122/c0bd6234-fe0d-4d77-a7f8-6cc02bc77907">



# 1. MATLAB app



https://github.com/tiagolins2/Image-Processing-Apps/assets/95873122/499a27ce-5a79-4eea-ac24-bd58fddee771



Running from the MATLAB Command Window (requires MATLAB installation, license needed)
1.	Prior to opening the image processing app on the command window, make sure that the current folder corresponds to the folder where the code is located at.

 ![tutorial_8_pro](https://github.com/tiagolins2/Image-Processing-Apps/assets/95873122/a8187ff5-e2b3-42e4-95aa-e6df1ab9da11)

2.	Next, write the following on the command window: Run_plate , or simply click open Run_plate.m file and press run
3.	Once open, check the settings for the type of plate you will run. For more information, you can click on the headings for a description of each setting. Ensure that:
a.	HSV color threshold settings are defined properly. The default ones may be from another run and could work well with your plate images
b.	Select settings on the check boxes
c.	Define plate dimensions
4.	You can then run one of the three protocols

Running as an executable (requires MATLAB runtime installation, no license needed)
1.	If you do not wish to have MATLAB installed in your computer, go to [MATLAB Runtime](https://www.mathworks.com/products/compiler/matlab-runtime.html) and install a version compatible with your operating system, with the runtime version number R2022a (9.12).  
2.	Double click on the file Run_plate.exe

 <img width="409" alt="tutorial_9_pro" src="https://github.com/tiagolins2/Image-Processing-Apps/assets/95873122/c6246ba8-8a24-46ce-ac01-f4b8c90e3e1f">

3.	A black window will appear. Keep it open whenever using the app, and wait for the main app to load
4.	
 <img width="468" alt="tutorial_10_pro" src="https://github.com/tiagolins2/Image-Processing-Apps/assets/95873122/436d4c70-f545-406f-9954-4a6d7b1f54a9">


4.	Once open, check that all settings correspond to what time of plate image you want to process. You can then run either one of the three protocols. For more information, you can click on the headings for a description of each setting.


Processing a single well plate
1.	To process a single plate image, first select the desired well plate dimension (i.e. 24) 
2.	Click on RUN SINGLE PLATE, as shown below:

 <img width="468" alt="tutorial_11_pro" src="https://github.com/tiagolins2/Image-Processing-Apps/assets/95873122/23c712d7-b0f4-4755-a35b-91f4634f20a9">

3.	You will be prompted to select an image. Make sure the selection is a raw image as shown below 
 
<img width="306" alt="tutorial_12_pro" src="https://github.com/tiagolins2/Image-Processing-Apps/assets/95873122/e9bb0f3d-1112-4956-af17-67f419392fc5">

4.	Next, a window will be displayed asking you to click anywhere on side you want to be selected. Here, I selected the plate on the left side (plate 1). Once you click, this window will automatically disappear. 

 <img width="264" alt="tutorial_13_pro" src="https://github.com/tiagolins2/Image-Processing-Apps/assets/95873122/a0f7ffac-dac9-46f9-81b1-0bd2e0ddfbdc">

5.	You can view the progress on the main app window. Wait until the app says completed. You can also click to view results on your computer
 
 <img width="468" alt="tutorial_14_pro" src="https://github.com/tiagolins2/Image-Processing-Apps/assets/95873122/7299a404-9612-4d68-af1e-0c1b39489470">

6.	You can continue processing other images or exit the app.


Updating the models: 
In case a batch of new images is taken, it may be necessary to update/re-train the machine learning models with new data for better accuracy.


Create new training sets:

Retraining the model: Although the model is written in python, retraining it should not require any previous Python knowledge. The training can be done in Google Colab. Follow the instructions for whichever model needs to be updated. Once done, download the model folder (.ckpt) and paste the unzipped version into the same directory as the python DuckPlate App. 

[96-well plate model](https://colab.research.google.com/drive/1SWK6kakSI3wPSGP0gQQB-E_I8S3qVDag?usp=drive_link)
[24-well plate model](https://colab.research.google.com/drive/15lob3aiShyZJG4l3DHmr5fbzVLEOz9SH?usp=drive_link)
[6-well plate model](https://colab.research.google.com/drive/1w2DEPqpE849efxu6ATihVbInGIgHKo0U?usp=drive_link)
[96-well plate duckweed segmentation model](https://colab.research.google.com/drive/1xtNG2CfiPLJ98sW9E3y6FqvPkstC6M1J?usp=drive_link)
[24-well plate duckweed segmentation model](https://colab.research.google.com/drive/1HeqZZpXBZMp_WpJcKre9nkv6bFAdhlaX?usp=drive_link)
[6-well plate duckweed segmentation model](https://colab.research.google.com/drive/1HeqZZpXBZMp_WpJcKre9nkv6bFAdhlaX?usp=drive_link)
[Plate classifier model]

