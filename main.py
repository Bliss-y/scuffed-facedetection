import tkinter as tk
from tkinter import *
from face_detection import faceDetection
from PIL import Image, ImageTk
from model import model
m = model.load('./oopmodel/', 'model2');


root = Tk()
text = "Click the Sign in! button for Attendance"
label = Label(root, text=text)
def cb(result):
    if not result == 'others' :
        img = Image.open('./temp/tmp.jpg')
        render = ImageTk.PhotoImage(img)
        img = Label(root, image=render)
        img.place(x=0, y=0)
        text = 'asdf'
        l = Label(root, text="Welcome " + result)
        l.pack()
        print(result);
        # text = "Welcome: " + text
        return 1;
label.pack()
signinbutton = Button(root, text="Sign In!",
            command=lambda :faceDetection.cap(model=m, cb=cb))
signinbutton.pack()
root.test = signinbutton
quit = Button(root, text="QUIT", command=root.destroy)
quit.pack()
# The following three commands are needed so the window pops
# up on top on Windows...
root.iconify()
root.update()
root.wm_title('Attendance :)')
root.deiconify()
root.mainloop()
