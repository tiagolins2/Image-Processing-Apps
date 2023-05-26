# Image-Processing-Apps

# 2. Python app
First open the application, either the standalone app (.exe) or the python app:

<img width="500" alt="tutorial_1_pro" src="https://github.com/tiagolins2/Image-Processing-Apps/assets/95873122/c5c17601-0cda-42ba-8819-d89b6ee8522d">

You can either choose one file or multiple. The files must be sized 4056x3040, with up to 2 well plates

<img width="608" alt="tutorial_2_pro" src="https://github.com/tiagolins2/Image-Processing-Apps/assets/95873122/d66481a0-fd4a-4d86-8e52-4b8dd0338f06">

Once you click Open, the main application will appear on the screen

<img width="600" alt="tutorial_3_pro" src="https://github.com/tiagolins2/Image-Processing-Apps/assets/95873122/6a8466b8-97bb-43f4-b0d6-095272e2f328">

Press Run. You will be able to see the progress of the image processing. Once completed, you can view the results by clicking the button Check results

<img width="600" alt="tutorial_4_pro" src="https://github.com/tiagolins2/Image-Processing-Apps/assets/95873122/c0bd6234-fe0d-4d77-a7f8-6cc02bc77907">



# 1. MATLAB app


Both apps can be run as standalone applications

Running Standalone
a. Double click on the .exe file
b. A black window will open. Do not close this window while running the app
c. Once the app is fully loaded, the following window will appear, prompting you to select a file to process
d. Click on the button, and select the file(s) you want to process (note that each image must be sized 4056x3040, and should contain two plates side by side. If one or both of the spots are empty, the app can still process the image)
e. 
DuckPlate app:

This app implements trained models to process well plate images


Update the models: 
In case a batch of new images is taken, it may be necessary to update the machine learning models with new data.


Create new training sets:

Retraining the model: Although the model is written in python, retraining it should not require any previous Python knowledge. The training can be done in Google Collab 
