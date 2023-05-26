from tkinter import *
from tkinter import messagebox

def everything():
    def Ok():
        uname = e1.get()
        password = e2.get()

        if(uname == "" and password == "") :
            messagebox.showinfo("", "Blank Not allowed")


        elif(uname == "Admin" and password == "123"):

            messagebox.showinfo("","Login Success")
            top.destroy()
            main.destroy()

        else :
            messagebox.showinfo("","Incorrent Username and Password")
        
    main = Tk()
    main.withdraw()
    top = Toplevel()
    top.title("Login")
    top.geometry("300x200")

    Label(top, text="UserName").place(x=10, y=10)
    Label(top, text="Password").place(x=10, y=40)

    e1 = Entry(top)
    e1.place(x=140, y=10)

    e2 = Entry(top)
    e2.place(x=140, y=40)
    e2.config(show="*")


    Button(top, text="Login", command=Ok ,height = 3, width = 13).place(x=10, y=100)
    return e1, e2;
    top.mainloop()



from tkinter import *
from tkinter import messagebox




root = Tk()
root.title("Main")
root.geometry("500x500")
Button(root, text='Apply',
                   command=everything).place(x=20, y=20)

Label(root, text="Welcome").place(x=10, y=10)
#everything()

root.mainloop()