import tkinter as tk
import customtkinter
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog, simpledialog 
from PIL import Image
from PIL import ImageTk as itk
from matplotlib import cm
#from model_fun import*
from import_models_v3 import*
import shutil
from scipy import ndimage
import cv2
import numpy as np;
import matplotlib.pyplot as plt

customtkinter.set_appearance_mode("dark")

class MainApplication():
    def __init__(self, mainframe, filez):
        super().__init__()
        #mainframe.withdraw()
        #splash = Splash(mainframe)
        #self.load_image() 
        mainframe.deiconify()
        mainframe.title('DuckPlate')
        mainframe.resizable(False, False)
        #mainframe.configure(background='#ffffff')
        self.filez = filez;
        
        self.plnum_val = 0;
        self.override_plnum = 0; 
        customtkinter.set_appearance_mode("dark")
        #self.style = ttk.Style()
        #self.style.configure('TFrame', background='#ffffff')
        #self.style.configure('TButton', background='#ffffff')
        #self.style.configure('TLabel', bverriackground='#ffffff', font=('Arial', 12))
        #self.style.configure('Header.TLabel', font=('Arial', 18, 'bold'))

        self.header_frame = ttk.Frame(mainframe)
        self.header_frame.pack()
        background = 'image20220213-162836.png'

	  #photo = itk.PhotoImage(file= background)
	  #canvas = tk.Canvas(root, width=500, height=500)         
        img=Image.open("dkplt2.png")#dkplt
        self.npimg = img;
        #img,dw=(import_model("Plate6_Wells.png"))
        self.model_96 = keras.models.load_model("96wp_model.ckpt",compile=False)
        self.model_6 = keras.models.load_model("6wp_model.ckpt",compile=False)
        self.model_24 = keras.models.load_model("24wp_model.ckpt",compile=False)
        self.model_plateid = keras.models.load_model("cp_plate_id_v2.ckpt",compile=False)
        #img = Image.fromarray(img)
        resized_image= img.resize((200,300))
        self.logo = itk.PhotoImage(resized_image)
        customtkinter.CTkLabel(self.header_frame, image=self.logo, text=" ").grid(row=1, column=1, rowspan=6, pady=10, padx=20)
          
        #customtkinter.CTkLabel(self.header_frame, text='    ', style='Header.TLabel').grid(row=0, column=1)
        self.h_min_slider = tk.Scale(self.header_frame, from_=0, to=179, orient=tk.HORIZONTAL, label="Hue Min", background='#ffffff')
        self.h_max_slider = tk.Scale(self.header_frame, from_=0, to=179, orient=tk.HORIZONTAL, label="Hue Max", background='#ffffff')
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
        self.apply = customtkinter.CTkButton(self.header_frame, text='Apply',
                   command=self.apply_threshold)
        self.green_thres_text=ttk.Label(self.header_frame, text='Green threshold:')
        #self.=ttk.Label(self.header_frame, text='Green threshold:')
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
        self.lower_content = ttk.Frame(mainframe)
        self.lower_content.pack()

        #ttk.Label(self.content_in_frame, text='Name:').grid(row=0, column=0, padx=5, sticky='sw')
        #ttk.Label(self.content_in_frame, text='Email:').grid(row=0, column=1, padx=5, sticky='sw')
        #ttk.Label(self.content_in_frame, text='Comments:').grid(row=2, column=0, padx=5, sticky='sw')

        #self.comment_name = ttk.Entry(self.content_in_frame, width=24, font=('Arial', 10))
        
        #self.comments = Text(self.content_in_frame, width=50, height=10, font=('Arial', 10))

        #self.comment_name.grid(row=1, column=0, padx=5)
       
        #self.comments.grid(row=3, column=0, columnspan=2, padx=5)
        
        self.submit=customtkinter.CTkButton(self.content_in_frame, text='Run',
                   command=self.upload_images, width=190,height=50, border_width=0, corner_radius=30).grid(row=1,column=0)
        #self.submit=customtkinter.CTkButton(self.content_in_frame, text='Select file',
        #           command=self.load_image).grid(row=8, column=0, padx=5, pady=5, sticky='e')
        #self.submit=customtkinter.CTkButton(self.content_in_frame, text='Select folder',
        #           command=self.upload_images).grid(row=8, column=1, padx=5, pady=5, sticky='e') 
        #self.change_view=customtkinter.CTkButton(self.content_in_frame, text='Change view',
        #           command=self.change_image).grid(row=8, column=2, padx=5, pady=5, sticky='w')
        #self.change_method=customtkinter.CTkButton(self.lower_content, text='Change Method',
        #           command=self.method_button)
        #self.change_method.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.settings=customtkinter.CTkButton(self.lower_content, text='Settings',
                   command=self.load_image)
        self.settings.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.comment = ttk.Label(self.lower_content, font=('Arial', 13), text=' ');
        self.comment.grid(row=2, column=0, padx=5, pady=5, sticky='w',columnspan=3);
      
        #self.link = ttk.Label(self.lower_content, text=" ", fg="blue", cursor="hand2") 
        #self.link.grid(row=3, column=1, padx=5, pady=5, sticky='w',columnspan=3);     

        #self.hide_button;
        #ttk.Button(self.content_in_frame, text='Adjust Segmentation',
        #           command=self.clear).grid(row=4, column=1, padx=5, pady=5, sticky='w')
        #splash.destroy()
        self.mainframe = mainframe; 
        ## show window again
        

    def submit(self):
        print(f'Name: {self.comment_name.get()}')
        print(f'Email: {self.comment_email.get()}')
        print(f'Comments: {self.comments.get(1.0, "end")}')
        self.clear()
        messagebox.showinfo(title='Comment info', message='Thanks for your comment!')
    
    def callback(self):
        webbrowser.open_new(self.url)

    def load_app(self):
       self.load = Tk()
       self.load.withdraw()
       self.top = Toplevel()
       self.top.title("Loading app")
       self.top.geometry("100x100") 
       def destroy(self):
           self.top.destroy()
           self.main.destroy()
       self.top.mainloop() 

    def open_folder(self):
       folder = filedialog.askopenfilename(initialdir = self.folder2check) 

 
    def load_image(self):
        
       def Ok(self):
         filepath = e1.get()
         plate_number = self.var2.get()
         #overide = e3.get()         
         self.filez = tuple([filepath]);
         #self.e2=plate_number
         self.plnum_val=variable.get()
         #print(f"{self.filez} - {type(self.filez)}");         

         main.withdraw()
         top.withdraw()
         top.destroy()
         main.destroy()
         #self.upload_image()

       def open_file(self):
          filepath = e1.get()
          filez = filedialog.askopenfilenames(title='Choose a file')
          e1.insert(0,filez) 

       def override(self):
          self.override_plnum=1;
          self.plnum_val=variable.get()

       main = Tk()
       main.withdraw()
       top = Toplevel()
       top.title("Login")
       top.geometry("350x200")

       Label(top, text="File path:").place(x=10, y=10)
       #Label(top, text = self.filez).place()
       Label(top, text="Plate number").place(x=10, y=60)
       OPTIONS = [
                  "0",
                  "6",
                  "24",
                  "96"   
                    ] 
       variable = StringVar()
       variable.set(OPTIONS[3]) # default value
       w = OptionMenu(top, variable, *OPTIONS) 
       w.place(x=220, y=110)
       b1 = Button(top, text="Select file",command= lambda: open_file(self))
       e1 = Entry(top)
       e1.delete(0, END); e1.insert(0,self.filez)
       e1.place(x=100, y=10)
       b1.place(x=280, y=10)
       #self.e1=e1;        
       
       OPTIONS_plate = [
                  "1",
                  "2",
                  "both" 
                    ]  
       self.var2 = StringVar()
       self.var2.set("Select a plate number - side")
       e2 = OptionMenu(top, self.var2, *OPTIONS_plate) 
       self.plate_number = self.var2.get()
 
       #e2 = Entry(top)
       e2.place(x=100, y=60)
       #self.e2=e2;
       self.var1=0;
       e3 = Checkbutton(top, text='Override plate classification',variable=self.var1, onvalue=1, offvalue=0, command= lambda: override(self))
       e3.place(x=50, y=110)
       self.e3=e3;
       
       
        
       Button(top, text="Apply", command= lambda: Ok(self) ,height = 3, width = 13).place(x=10, y=140)
       top.mainloop() 

    def clear(self):
        self.comment_name.delete(0, 'end')
        self.comment_email.delete(0, 'end')
        self.comments.delete(1.0, 'end')
    


    def upload_image(self):
      # Open a file dialog to select an image
      # file_path = filedialog.askopenfilename()
      filez = self.filez;
      filez = filez[0];   
      #filez = filedialog.askopenfilenames(title='Choose a file')
      print(filez)
      file_path=os.path.basename(filez); print(file_path)
      img_dir=file_path[0:-4]; print(img_dir)
      filez=os.path.dirname(filez); print(filez)
      print(filez)
      image_splitter(filez,file_path) 
      #ijk = simpledialog.askinteger('Select plate number:','plate');
      #ijk = self.e2;
      file_path=f"{filez}\\data\\{img_dir}\\{img_dir}_plate{ijk}.jpg"
      print(f"file_path= {file_path}") 
      #filez = filedialog.askdirectory(title='Choose a file')
      self.comment["text"] = "Opening..." 
      # Check if a file was selected
      #for file_path in filez:  
      if file_path:
            print("start processing");
            #mainframe.update() 
            # Open the image using PIL
            #image = Image.open(file_path)
            if self.override_plnum==1:
               plnum = int(self.plnum_val)
               print(f"using {plnum} well plate dimensions")   
            else: 
               plnum=import_model_identify_plate(file_path,self.model_plateid);
            
            self.comment["text"] = "Processing images..."  
            ##temp:
            if plnum==0:
               plnum=0;
            if plnum==96:
               #img,self.dw=(import_model(file_path,model_96,model_plateid,plnum,([12, 8, 0.00009, 3.20574346e+02, 3.90657484e+02, 50, 50, 50, 15,192,195])))
               img,self.dw,area=(import_model(file_path,self.model_96,plnum,([12, 8, 0.00009, 3.20574346e+02, 3.90657484e+02, 50, 50, 50, 15,192,195]))) 
               img_new = Image.fromarray(img*255)
               img_raw = plt.imread(file_path);
               #img_raw=cv2.cvtColor(np.array(img_raw), cv2.COLOR_HSV2RGB) 
               img_raw=resize_image(img_raw,(672, 448)) 
               kernel = np.ones((10, 10), np.uint8)
               img_dilation = cv2.dilate(img, kernel, iterations=1) 
               edges = img_dilation - img;
               img_raw[:,:,0] = np.squeeze(img_raw[:,:,0])*(1-edges) 
               img_raw[:,:,1] = np.squeeze(img_raw[:,:,1])*(1-edges) 
               img_raw[:,:,2] = np.squeeze(img_raw[:,:,2])*(1-edges)
                  
               if (img_raw.max())>1:
                  img_raw=Image.fromarray((img_raw).astype(np.uint8));
               else:
                  img_raw=Image.fromarray((img_raw*255).astype(np.uint8));  
               self.npimg = img_raw;
               resized_image= img_new.resize((200,300))
               resized_image_raw= img_raw.resize((200,300))
               self.npraw = resized_image_raw;     
               #resized_image_raw[:,:,0]=resized_image_raw[:,:,0]+resized_image  
               self.logo = itk.PhotoImage(resized_image_raw)
               self.mask = itk.PhotoImage(resized_image)
               #self.mask = cv2.Canny(resized_image,100,200);
               self.view_img=1;
               ttk.Label(self.header_frame, image=self.logo).grid(row=1, column=1, rowspan=6)
            if plnum==6:
               img,self.dw,area=(import_model(file_path,self.model_6,plnum,([3, 2, 0.00009, 3.20574346e+02, 3.90657484e+02, 182, 150, 150, 90,192*4,831]))) ### need to change the last value of dw_size
               img_new = Image.fromarray(img*255)
               img_raw = plt.imread(file_path);
               #img_raw=cv2.cvtColor(np.array(img_raw), cv2.COLOR_HSV2RGB) 
               img_raw=resize_image(img_raw,(672, 448)) 
               kernel = np.ones((30, 30), np.uint8)
               img_dilation = cv2.dilate(img, kernel, iterations=1) 
               edges = img_dilation - img;
               img_raw[:,:,0] = np.squeeze(img_raw[:,:,0])*(1-edges) 
               img_raw[:,:,1] = np.squeeze(img_raw[:,:,1])*(1-edges) 
               img_raw[:,:,2] = np.squeeze(img_raw[:,:,2])*(1-edges)
               if (img_raw.max())>1:
                  img_raw=Image.fromarray((img_raw).astype(np.uint8));
               else:
                  img_raw=Image.fromarray((img_raw*255).astype(np.uint8));  
               #img_raw=Image.fromarray((img_raw).astype(np.uint8));
               self.npimg = img_raw;
               resized_image= img_new.resize((200,300))
               resized_image_raw= img_raw.resize((200,300))
               self.npraw = resized_image_raw;     
               #resized_image_raw[:,:,0]=resized_image_raw[:,:,0]+resized_image  
               self.logo = itk.PhotoImage(resized_image_raw)
               self.mask = itk.PhotoImage(resized_image)
               #self.mask = cv2.Canny(resized_image,100,200);
               self.view_img=1;
               ttk.Label(self.header_frame, image=self.logo).grid(row=1, column=1, rowspan=6)

            if plnum==24:
                                                         #rows, columns, p1_rho, Lx, Ly, dist_circ, indx, indy, R,dw_size,dw_size_raw
               img,self.dw,area=(import_model(file_path,self.model_24,plnum,([6, 4, 0.00009, 3.320e+02, 3.85e+02, 95, 90, 90, 30,320,383]))) ### need to change the last value of dw_size
               img_new = Image.fromarray(img*255)
               img_raw = plt.imread(file_path);
               #img_raw=cv2.cvtColor(np.array(img_raw), cv2.COLOR_HSV2RGB) 
               img_raw=resize_image(img_raw,(672, 448)) 
               kernel = np.ones((30, 30), np.uint8)
               img_dilation = cv2.dilate(img, kernel, iterations=1) 
               edges = img_dilation - img;
               img_raw[:,:,0] = np.squeeze(img_raw[:,:,0])*(1-edges) 
               img_raw[:,:,1] = np.squeeze(img_raw[:,:,1])*(1-edges) 
               img_raw[:,:,2] = np.squeeze(img_raw[:,:,2])*(1-edges)
               if (img_raw.max())>1:
                  img_raw=Image.fromarray((img_raw).astype(np.uint8));
               else:
                  img_raw=Image.fromarray((img_raw*255).astype(np.uint8));  
               #img_raw=Image.fromarray((img_raw).astype(np.uint8));
               self.npimg = img_raw;
               resized_image= img_new.resize((200,300))
               resized_image_raw= img_raw.resize((200,300))
               self.npraw = resized_image_raw;     
               #resized_image_raw[:,:,0]=resized_image_raw[:,:,0]+resized_image  
               self.logo = itk.PhotoImage(resized_image_raw)
               self.mask = itk.PhotoImage(resized_image)
               #self.mask = cv2.Canny(resized_image,100,200);
               self.view_img=1;
               ttk.Label(self.header_frame, image=self.logo).grid(row=1, column=1, rowspan=6)
            
            if self.methodidx==2:
              self.comment["text"] = "Segmenting images"
              list_wellsv=list_wells(plnum)
              data=[];
              for i in range(len(self.dw)):
                 raw_img=self.dw[i]; 
                 segmented_well_img,area=segment_wells(raw_img,self.h_max,self.h_min,self.s_max,self.s_min,self.v_max,self.v_min,self.GT,1)
                 data.append(area); 
                 ###commented to reduce memory
                 plt.imsave(f"{file_path[0:-4]}/{list_wellsv[i]}_mask.png", segmented_well_img, cmap='gray')
              np.savetxt(f"{file_path[0:-4]}/data.csv", data, delimiter=",")    
            
            #messagebox.showinfo(title='Scanning plate', message=f"plate at {file_path} has been scanned")
		
      else:
            messagebox.showinfo("No file was selected.")
      self.comment["text"] = "Completed"


    def upload_images(self):
      # Open a file dialog to select an image
      # file_path = filedialog.askopenfilename()
      #filez = filedialog.askopenfilenames(title='Choose a file')
      
      #filez = filedialog.askdirectory(title='Choose a folder containing images')
      print(f"{self.filez} - {type(self.filez)}");
      filez = self.filez
      files=[];
      for fname in filez:
       if os.path.isdir(fname):
         pass
       else:
         files.append(fname) 

      #files = os.listdir(filez)
      print(files)
      #files=os.listdir(filez)
      self.comment["text"] = "Opening..." 
      self.mainframe.update()
      # Check if a file was selected
      #basedir = os.path.basename(filez);
      for file_path in files:
       basedir, filename = os.path.split(file_path)
       #split image into two:
       self.comment["text"] = f"Opening {filename}" 
       self.mainframe.update()
       img_dir=filename[0:-4]; 
       image_splitter(basedir,filename)
       self.basedir = basedir; 
       self.filename = filename; 
       for ijk in [1,2]:
          img_dir=filename[0:-4];
          file_path=f"data\\{img_dir}\\{img_dir}_plate{ijk}.jpg"
          self.comment["text"] = f"Processing {img_dir}_plate{ijk}.jpg" 
          self.mainframe.update()  
          if file_path:
            #mainframe.update() 
            # Open the image using PIL
            #image = Image.open(file_path)
            img_raw = plt.imread(basedir+'\\'+file_path);
            customtkinter.CTkLabel(self.header_frame, image=itk.PhotoImage((Image.fromarray(img_raw)).resize((200,300))), text=" ").grid(row=1, column=0, rowspan=6, padx=20, pady=10);
            customtkinter.CTkLabel(self.header_frame,  width=140, height=200, text=" ").grid(row=1, column=1, rowspan=6, padx=20, pady=10);
            customtkinter.CTkLabel(self.header_frame,  width=140, height=200, text=" ").grid(row=1, column=2, rowspan=6, padx=20, pady=10);
            self.mainframe.update();   
            print("##########################################################################################################################")
            print(f"Starting processing for {file_path}")
            if self.override_plnum==1:
               plnum = int(self.plnum_val)
               print(f"using {plnum} well plate dimensions")   
            else: 
               plnum=import_model_identify_plate(basedir+'\\'+file_path,self.model_plateid);
            
            self.comment["text"] = "Processing images..."  
            if plnum==96:
               #img,self.dw=(import_model(basedir+'\\'+file_path,"cp_96plate.ckpt",plnum,([12, 8, 0.00009, 3.20574346e+02, 3.90657484e+02, 50, 50, 50, 15,192,195]),self.h_min,self.h_max,self.s_min,self.s_max,self.v_min,self.v_max,self.GT))
               img,self.dw,area=(import_model(basedir+'\\'+file_path,self.model_96,plnum,([12, 8, 0.00009, 3.20574346e+02, 3.90657484e+02, 50, 50, 50, 15,192,195]))) 
            if plnum==6:
               img,self.dw,area=(import_model(basedir+'\\'+file_path,self.model_6,plnum,([3, 2, 0.00009, 3.20574346e+02, 3.90657484e+02, 182, 150, 150, 90,750,750]))) ### need to change the last value of dw_size
            if plnum==24:
               img,self.dw,area=(import_model(basedir+'\\'+file_path,self.model_24,plnum,([6, 4, 0.00009, 3.320e+02, 3.85e+02, 95, 90, 90, 30,320,383]))) ### need to change the last value of dw_size
            if plnum==0:
              try:
                # remove the directory using the rmdir() method
                shutil.rmtree(basedir+'\\'+file_path[0:-4])
                print("Previous directory has been deleted successfully!")
              except OSError as error:
                print(f"Error: {file_path[0:-4]} : {error.strerror}")
            img_new = Image.fromarray(img*255)
            #img_raw = plt.imread(basedir+'\\'+file_path);
            img_raw=resize_image(img_raw,(672, 448)) 
            kernel = np.ones((10, 10), np.uint8)
            img_dilation = cv2.dilate(img, kernel, iterations=1) 
            edges = img_dilation - img;
            #img_raw0 = img_raw; img_raw0 = Image.fromarray(img_raw0)
            img_raw[:,:,0] = np.squeeze(img_raw[:,:,0])*(1-edges) 
            img_raw[:,:,1] = np.squeeze(img_raw[:,:,1])*(1-edges) 
            img_raw[:,:,2] = np.squeeze(img_raw[:,:,2])*(1-edges)
                  
            if (img_raw.max())>1:
                img_raw=Image.fromarray((img_raw).astype(np.uint8));
            else:
                img_raw=Image.fromarray((img_raw*255).astype(np.uint8));  
            self.npimg = img_raw;
            resized_image= img_new.resize((200,300))
            resized_image_raw= img_raw.resize((200,300))
            self.npraw = resized_image_raw;     
            
            self.logo = itk.PhotoImage(resized_image_raw)
            self.mask = itk.PhotoImage(resized_image)
            self.view_img = 1;
            self.folder2check = basedir+'\\data'
            self.acess_results = ttk.Button(self.lower_content, text ="Check results", command = self.open_folder)
            self.acess_results.grid(row=3, column=1, rowspan=6, padx=20, pady=10, ipady=10, ipadx=10);
 
            customtkinter.CTkLabel(self.header_frame, image=self.mask, text=" ").grid(row=1, column=1, rowspan=6, padx=20, pady=10);
            customtkinter.CTkLabel(self.header_frame, image=self.logo, text=" ").grid(row=1, column=2, rowspan=6, padx=20, pady=10);
            self.mainframe.update()   
            
            
            if self.methodidx==2:
              self.comment["text"] = "Segmenting images"
              data=[];
              for i in range(len(self.dw)):
                 raw_img=self.dw[i]; 
                 segmented_well_img,area=segment_wells(raw_img,self.h_max,self.h_min,self.s_max,self.s_min,self.v_max,self.v_min,self.GT,1)
                 data.append(area); 
                 ###commented to reduce memory
                 plt.imsave(f"{basedir}/{file_path[0:-4]}/{list_wellsv[i]}_mask.png", segmented_well_img, cmap='gray')
              np.savetxt(f"{basedir}/{file_path[0:-4]}/data.csv", data, delimiter=",")    
            #self.npraw = resized_image_raw;     
            #self.logo = itk.PhotoImage(resized_image_raw)
            #self.mask = itk.PhotoImage(resized_image)
            self.view_img=1;
            self.comment["text"] = f"Plate at {file_path} has been scanned" 
            #self.mainframe.update()    
          else:
            messagebox.showinfo("No file was selected.")
      self.comment["text"] = "Completed"
      print("##########################################################################################################################")
      print("Completed!")

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
         self.change_method["text"]="Change to ML"  
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
      #self.change_method=customtkinter.CTkButton(self.content_in_frame, text='Change Method',
      #             command=self.show_button).grid(row=7, column=2, padx=5, pady=5, sticky='w')
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
       print(np.max(hsv_image[:,:,0]))
       print(np.max(hsv_image[:,:,1]))
       print(np.max(hsv_image[:,:,2]))
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
       img,area = segment_wells(img,self.h_max,self.h_min,self.s_max,self.s_min,self.v_max,self.v_min,self.GT,0)
       #image_array=normalize_saturation(image_array);
       #img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
       #img[:,:,0] = np.squeeze(image_array[:,:,0])*thresholded_image2
       #img[:,:,1] = np.squeeze(image_array[:,:,1])*thresholded_image2
       #img[:,:,2] = np.squeeze(image_array[:,:,2])*thresholded_image2
       #img = cv2.bitwise_and(hsv_image,hsv_image,mask = (thresholded_image2))
       #print(img.shape)
       #img = Image.fromarray(thresholded_image)
       img = Image.fromarray(img)
       resized_image= img.resize((200,300))
       self.logo = itk.PhotoImage(resized_image)
       ttk.Label(self.header_frame, image=self.logo).grid(row=1, column=1, rowspan=6)


class StartupWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("Welcome")
        ico = Image.open('duckplate_icon2.png')
        photo = itk.PhotoImage(ico)
        self.master.wm_iconphoto(True, photo)

        # Create a button that will launch the main application
        self.button = customtkinter.CTkButton(master, text="What images do you want to process?", command=self.select_image, width=190,height=50, border_width=0, corner_radius=30)
        self.button.place(relx=0.5, rely=0.5, anchor=tk.CENTER);
        self.master.geometry("400x300")
        
    def select_image(self):
        self.filez = filedialog.askopenfilenames(title='Choose a file')
        self.launch_main_app()
          
    def launch_main_app(self):
        # Hide the startup window
        self.master.withdraw()
        
        # Create the main application window
        self.main_app = tk.Toplevel(self.master)
        self.main_app.title("Main Application")
        self.main_app.geometry('850x600')
        ico = Image.open('duckplate_icon2.png')
        photo = itk.PhotoImage(ico)
        self.main_app.wm_iconphoto(True, photo)
        print(f"{self.filez} - {type(self.filez)}");
        # Initialize the main application
        self.app = MainApplication(self.main_app, self.filez)

if __name__ == '__main__':
    root = customtkinter.CTk()
    ico = Image.open('duckplate_icon2.png')
    photo = itk.PhotoImage(ico)
    root.wm_iconphoto(True, photo)
    app = StartupWindow(root)
    root.mainloop()
