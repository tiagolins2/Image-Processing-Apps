import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog 
from PIL import Image
from PIL import ImageTk as itk
from matplotlib import cm
#from model_fun import*
from import_models_v2 import*

class Feedback(tk.Frame):

    def __init__(self, mainframe):
        mainframe.title('DuckPlate')
        mainframe.resizable(False, False)
        mainframe.configure(background='#ffffff')

        self.style = ttk.Style()
        self.style.configure('TFrame', background='#ffffff')
        self.style.configure('TButton', background='#ffffff')
        self.style.configure('TLabel', background='#ffffff', font=('Arial', 12))
        self.style.configure('Header.TLabel', font=('Arial', 18, 'bold'))

        self.header_frame = ttk.Frame(mainframe)
        self.header_frame.pack()
        background = 'image20220213-162836.png'

	  #photo = itk.PhotoImage(file= background)
	  #canvas = tk.Canvas(root, width=500, height=500)         
        img=Image.open("duckplate.png")
        self.npimg = img;
        #img,dw=(import_model("Plate6_Wells.png"))
        
        #img = Image.fromarray(img)
        resized_image= img.resize((200,300))
        self.logo = itk.PhotoImage(resized_image)
        ttk.Label(self.header_frame, image=self.logo).grid(row=1, column=1, rowspan=6)
        ttk.Label(self.header_frame, text='    ', style='Header.TLabel').grid(row=0, column=1)
        self.h_min_slider = tk.Scale(self.header_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="Hue Min", background='#ffffff')
        self.h_max_slider = tk.Scale(self.header_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="Hue Max", background='#ffffff')
        self.s_min_slider = tk.Scale(self.header_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="Saturation Min", background='#ffffff')
        self.s_max_slider = tk.Scale(self.header_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="Saturation Max", background='#ffffff')
        self.v_min_slider = tk.Scale(self.header_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="Value Min", background='#ffffff')
        self.v_max_slider = tk.Scale(self.header_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="Value Max", background='#ffffff')
        self.h_min_slider.grid(row=1, column=3, padx=50, sticky='s')
        self.h_max_slider.grid(row=2, column=3, padx=50, sticky='s')
        self.s_min_slider.grid(row=3, column=3, padx=50, sticky='s')
        self.s_max_slider.grid(row=4, column=3, padx=50, sticky='s')
        self.v_min_slider.grid(row=5, column=3, padx=50, sticky='s')
        self.v_max_slider.grid(row=6, column=3, padx=50, sticky='s')
        self.apply = ttk.Button(self.header_frame, text='Apply',
                   command=self.apply_threshold)
        self.green_thres_text=ttk.Label(self.header_frame, text='Green threshold:')
        self.green_thres_text.grid(row=4, column=4, padx=5, sticky='sw')
        self.green_thres_value = ttk.Entry(self.header_frame, width=24, font=('Arial', 10))
        self.green_thres_value.grid(row=5, column=4, padx=5) 
        self.apply.grid(row=1, column=4, padx=50, rowspan=3)
        self.h_min_slider.grid_forget()
        self.h_max_slider.grid_forget()
        self.v_min_slider.grid_forget()
        self.v_max_slider.grid_forget()
        self.s_min_slider.grid_forget()
        self.s_max_slider.grid_forget()
        self.apply.grid_forget()
        self.green_thres_text.grid_forget()
        self.green_thres_value.grid_forget()
        self.methodidx = 1;

        #
        #ttk.Label(self.header_frame, wraplength=300,
        #          text=(
        #              'Add your name, email, and comment, then click submit to add your comment.  Click clear if you make a mistake.')).grid(
        #    row=1, column=1)

        self.content_in_frame = ttk.Frame(mainframe)
        self.content_in_frame.pack()

        #ttk.Label(self.content_in_frame, text='Name:').grid(row=0, column=0, padx=5, sticky='sw')
        #ttk.Label(self.content_in_frame, text='Email:').grid(row=0, column=1, padx=5, sticky='sw')
        #ttk.Label(self.content_in_frame, text='Comments:').grid(row=2, column=0, padx=5, sticky='sw')

        #self.comment_name = ttk.Entry(self.content_in_frame, width=24, font=('Arial', 10))
        self.comment = ttk.Label(self.content_in_frame, width=24, font=('Arial', 10), text=' ')
        #self.comments = Text(self.content_in_frame, width=50, height=10, font=('Arial', 10))

        #self.comment_name.grid(row=1, column=0, padx=5)
        self.comment.grid(row=7, column=0, rowspan=3)
        #self.comments.grid(row=3, column=0, columnspan=2, padx=5)

        self.submit=ttk.Button(self.content_in_frame, text='Select file',
                   command=self.upload_image).grid(row=8, column=0, padx=5, pady=5, sticky='e')
        self.submit=ttk.Button(self.content_in_frame, text='Select folder',
                   command=self.upload_images).grid(row=8, column=1, padx=5, pady=5, sticky='e') 
        self.change_view=ttk.Button(self.content_in_frame, text='Change view',
                   command=self.change_image).grid(row=8, column=2, padx=5, pady=5, sticky='w')
        self.change_method=ttk.Button(self.content_in_frame, text='Change Method',
                   command=self.method_button)
        self.change_method.grid(row=8, column=3, padx=5, pady=5, sticky='w')
        #self.hide_button;
        #ttk.Button(self.content_in_frame, text='Adjust Segmentation',
        #           command=self.clear).grid(row=4, column=1, padx=5, pady=5, sticky='w')

    def submit(self):
        print(f'Name: {self.comment_name.get()}')
        print(f'Email: {self.comment_email.get()}')
        print(f'Comments: {self.comments.get(1.0, "end")}')
        self.clear()
        messagebox.showinfo(title='Comment info', message='Thanks for your comment!')

    def clear(self):
        self.comment_name.delete(0, 'end')
        self.comment_email.delete(0, 'end')
        self.comments.delete(1.0, 'end')
    
    def upload_image(self):
      # Open a file dialog to select an image
      # file_path = filedialog.askopenfilename()
      filez = filedialog.askopenfilenames(title='Choose a file')
      #filez = filedialog.askdirectory(title='Choose a file')
      self.comment["text"] = "Opening..." 
      # Check if a file was selected
      for file_path in filez:  
        if file_path:
            #mainframe.update() 
            # Open the image using PIL
            #image = Image.open(file_path)
            plnum=import_model_identify_plate(file_path);
            self.comment["text"] = "Processing images..."  
            if plnum==96:
               #img,self.dw=(import_model(file_path,"cp_96plate.ckpt",plnum,([12, 8, 0.00009, 3.20574346e+02, 3.90657484e+02, 50, 50, 50, 15,192,195])))
               img,self.dw=(import_model(file_path,"cpnew96_model2.ckpt",plnum,([12, 8, 0.00009, 3.20574346e+02, 3.90657484e+02, 50, 50, 50, 15,192,195]))) 
            if plnum==6:
               img,self.dw=(import_model(file_path,"cp6wp_v3.ckpt",plnum,([3, 2, 0.00009, 3.20574346e+02, 3.90657484e+02, 182, 150, 150, 90,750,750]))) ### need to change the last value of dw_size
            img = Image.fromarray(img*255)
            img_raw=Image.open(file_path)
            self.npimg = img_raw;
            resized_image= img.resize((200,300))
            resized_image_raw= img_raw.resize((200,300))
            
            
            if self.methodidx==2:
              self.comment["text"] = "Segmenting images"
              list_wellsv=list_wells(plnum)
              data=[];
              for i in range(len(self.dw)):
                 raw_img=self.dw[i]; 
                 segmented_well_img,area=segment_wells(raw_img,self.h_max,self.h_min,self.s_max,self.s_min,self.v_max,self.v_min,self.GT)
                 data.append(area); 
                 ###commented to reduce memory
                 #plt.imsave(f"{file_path[0:-4]}/{list_wellsv[i]}.png", segmented_well_img, cmap='gray')
              np.savetxt(f"{file_path[0:-4]}/data.csv", data, delimiter=",")    
            self.npraw = resized_image_raw;     
            self.logo = itk.PhotoImage(resized_image_raw)
            self.mask = itk.PhotoImage(resized_image)
            self.view_img=1;
            ttk.Label(self.header_frame, image=self.logo).grid(row=1, column=1, rowspan=6)
            #messagebox.showinfo(title='Scanning plate', message=f"plate at {file_path} has been scanned")
		
        else:
            messagebox.showinfo("No file was selected.")
      self.comment["text"] = "Completed"


    def upload_images(self):
      # Open a file dialog to select an image
      # file_path = filedialog.askopenfilename()
      #filez = filedialog.askopenfilenames(title='Choose a file')
      filez = filedialog.askdirectory(title='Choose a folder containing images')
      files = filter(os.path.isfile, os.listdir(filez) )
      #files=os.listdir(filez)
      self.comment["text"] = "Opening..." 
      # Check if a file was selected
      for file_path in files:  
        if file_path:
            #mainframe.update() 
            # Open the image using PIL
            #image = Image.open(file_path)
            plnum=import_model_identify_plate(file_path);
            self.comment["text"] = "Processing images..."  
            if plnum==96:
               #img,self.dw=(import_model(filez+'\\'+file_path,"cp_96plate.ckpt",plnum,([12, 8, 0.00009, 3.20574346e+02, 3.90657484e+02, 50, 50, 50, 15,192,195]),self.h_min,self.h_max,self.s_min,self.s_max,self.v_min,self.v_max,self.GT))
               img,self.dw=(import_model(filez+'\\'+file_path,"cpnew96_model2.ckpt",plnum,([12, 8, 0.00009, 3.20574346e+02, 3.90657484e+02, 50, 50, 50, 15,192,195]))) 
            if plnum==6:
               img,self.dw=(import_model(filez+'\\'+file_path,"cp6wp_v3.ckpt",plnum,([3, 2, 0.00009, 3.20574346e+02, 3.90657484e+02, 182, 150, 150, 90,750,750]))) ### need to change the last value of dw_size
            img = Image.fromarray(img*255)
            img_raw=Image.open(filez+'\\'+file_path)
            self.npimg = img_raw;
            resized_image= img.resize((200,300))
            resized_image_raw= img_raw.resize((200,300))
            
            
            if self.methodidx==2:
              self.comment["text"] = "Segmenting images"
              data=[];
              for i in range(len(self.dw)):
                 raw_img=self.dw[i]; 
                 segmented_well_img, area=segment_wells(raw_img,self.h_max,self.h_min,self.s_max,self.s_min,self.v_max,self.v_min,self.GT)
                 data.append(area);
                 ###commented to reduce memory 
                 #plt.imsave(f"{filez}\\{file_path[0:-4]}/well_{i}.png", segmented_well_img)
              np.savetxt(f"{filez}\\{file_path[0:-4]}/{file_path[0:-4]}_data.csv", data, delimiter=",")  
            self.npraw = resized_image_raw;     
            self.logo = itk.PhotoImage(resized_image_raw)
            self.mask = itk.PhotoImage(resized_image)
            self.view_img=1;
            ttk.Label(self.header_frame, image=self.logo).grid(row=1, column=1, rowspan=6)
            #messagebox.showinfo(title='Scanning plate', message=f"plate at {file_path} has been scanned")
		
        else:
            messagebox.showinfo("No file was selected.")
      self.comment["text"] = "Completed"

    def change_image(self):
            if self.view_img==1:
              ttk.Label(self.header_frame, image=self.mask).grid(row=1, column=1, rowspan=6)
              self.view_img=2
            elif self.view_img==2:
              ttk.Label(self.header_frame, image=self.logo).grid(row=1, column=1, rowspan=6)
              self.view_img=1
   
    def method_button(self):
      


      
      
      if self.methodidx==1:
         self.methodidx=2 
         self.change_method["text"]="Change to MLT"  
         self.h_min_slider.grid(row=1, column=3, padx=50, sticky='s')
         self.h_max_slider.grid(row=2, column=3, padx=50, sticky='s')
         self.s_min_slider.grid(row=3, column=3, padx=50, sticky='s')
         self.s_max_slider.grid(row=4, column=3, padx=50, sticky='s')
         self.v_min_slider.grid(row=5, column=3, padx=50, sticky='s')
         self.v_max_slider.grid(row=6, column=3, padx=50, sticky='s')
         self.apply.grid(row=1, column=4, padx=50,rowspan=3)
         self.green_thres_text.grid(row=4, column=4, padx=5, sticky='sw')
         self.green_thres_value.grid(row=5, column=4, padx=5) 
      elif self.methodidx==2:
         self.methodidx=1
         self.change_method["text"]="Change to HSV" 
         self.h_min_slider.grid_forget()
         self.h_max_slider.grid_forget()
         self.v_min_slider.grid_forget()
         self.v_max_slider.grid_forget()
         self.s_min_slider.grid_forget()
         self.s_max_slider.grid_forget()
         self.apply.grid_forget()
         self.green_thres_text.grid_forget()
         self.green_thres_value.grid_forget()

    def hide_button(self):
      # This will remove the widget from toplevel
      #self.change_method.grid_remove() 
      #self.change_method=ttk.Button(self.header_frame, text='Change to HSV',
      #             command=self.show_button)
      #self.change_method.grid(row=7, column=2, padx=5, pady=5, sticky='w')
      self.change_method["text"]="Change to HSV" 
      self.change_method["command"]=show_button(self)
      self.h_min_slider.grid_forget()
      self.h_max_slider.grid_forget()
      self.v_min_slider.grid_forget()
      self.v_max_slider.grid_forget()
      self.s_min_slider.grid_forget()
      self.s_max_slider.grid_forget()
      self.apply.grid_forget()
      self.green_thres_text.grid_forget()
      self.green_thres_value.grid_forget()
  
    # Method to make Button(widget) visible 
    def show_button(self):
      # This will recover the widget from toplevel
      #self.change_method.grid_remove()
      #self.change_method=ttk.Button(self.header_frame, text='Change to MLT',
      #             command=self.hide_button)
      #self.change_method.grid(row=7, column=2, padx=5, pady=5, sticky='w')
      self.change_method["text"]="Change to MLT" 
      self.change_method["command"]=hide_button(self) 
      self.h_min_slider.grid(row=1, column=3, padx=50, sticky='s')
      self.h_max_slider.grid(row=2, column=3, padx=50, sticky='s')
      self.s_min_slider.grid(row=3, column=3, padx=50, sticky='s')
      self.s_max_slider.grid(row=4, column=3, padx=50, sticky='s')
      self.v_min_slider.grid(row=5, column=3, padx=50, sticky='s')
      self.v_max_slider.grid(row=6, column=3, padx=50, sticky='s')
      self.apply.grid(row=1, column=4, padx=50,rowspan=3)
      self.change_method=ttk.Button(self.content_in_frame, text='Change Method',
                   command=self.show_button).grid(row=7, column=2, padx=5, pady=5, sticky='w')
      self.green_thres_text.grid(row=4, column=4, padx=5, sticky='sw')
      self.green_thres_value.grid(row=5, column=4, padx=5) 

    def applyGT_threshold(self):
       image_array=np.array(self.npimg);
       red_channel = image_array[:, :, 0]
       green_channel = image_array[:, :, 1]
       blue_channel = image_array[:, :, 2]

       # Calculate the green intensity of each pixel relative to the red and blue values
       green_intensity = (255*np.ones([len(image_array[0]),len(image_array[1])])+green_channel - ((red_channel)/2+(blue_channel)/2))/2
       # np.divide(green_channel, (red_channel + blue_channel), out=np.zeros_like(green_channel), where=(red_channel + blue_channel) != 0)
       GT_mask=green_intensity>self.GT
       return GT_mask
    
    def apply_threshold(self):
       self.h_min = self.h_min_slider.get()
       self.h_max = self.h_max_slider.get()
       self.s_min = self.s_min_slider.get()
       self.s_max = self.s_max_slider.get()
       self.v_min = self.v_min_slider.get()
       self.v_max = self.v_max_slider.get()
       self.GT = self.green_thres_value.get()
       # Define the lower and upper color threshold values as numpy arrays
       lower_color = np.array([self.h_min, self.s_min, self.v_min])
       upper_color = np.array([self.h_max, self.s_max, self.v_max])
       hsv_image = cv2.cvtColor(np.array(self.npimg), cv2.COLOR_RGB2HSV)
       # Apply the color thresholding to the image
       thresholded_image = cv2.inRange(hsv_image, lower_color, upper_color)
       #thresholded_image=np.expand_dims(thresholded_image, axis=-1)
       print(thresholded_image.shape)
       print(type(thresholded_image))
       #thresholded_image = applyGT_threshold(self)
       image_array=np.array(self.npimg);
       red_channel = image_array[:, :, 0]
       green_channel = image_array[:, :, 1]
       blue_channel = image_array[:, :, 2]
       # Calculate the green intensity of each pixel relative to the red and blue values
       green_intensity = (255 + green_channel - ((red_channel)*0.5 + (blue_channel)*0.5))*0.5
       thresholded_image2 = (green_intensity>float(self.GT))*1
     
       #thresholded_image=np.expand_dims(thresholded_image, axis=-1)
       print(thresholded_image2.shape)
       print(type(thresholded_image2))
       print(hsv_image.shape)
       
       #print(type(thresholded_image))
       thresholded_image2=thresholded_image2+thresholded_image
       thresholded_image2=(thresholded_image2>=1)*1
       img = image_array;
       #img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
       img[:,:,0] = np.squeeze(image_array[:,:,0])*thresholded_image2
       img[:,:,1] = np.squeeze(image_array[:,:,1])*thresholded_image2
       img[:,:,2] = np.squeeze(image_array[:,:,2])*thresholded_image2
       #img = cv2.bitwise_and(hsv_image,hsv_image,mask = (thresholded_image2))
       print(img.shape)
       #img = Image.fromarray(thresholded_image)
       img = Image.fromarray(img)
       resized_image= img.resize((200,300))
       self.logo = itk.PhotoImage(resized_image)
       ttk.Label(self.header_frame, image=self.logo).grid(row=1, column=1, rowspan=6)

def main():
    
    root = Tk()
    ico = Image.open('duckplate_icon2.png')
    photo = itk.PhotoImage(ico)
    root.wm_iconphoto(True, photo)
    root.geometry('700x500')
    feedback = Feedback(mainframe=root)
    root.mainloop()



if __name__ == '__main__': main()