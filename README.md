# Image-Processing-Apps


1. MATLAB app

2. Python app
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
