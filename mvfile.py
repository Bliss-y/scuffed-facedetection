import os
i=0
dir = './images/train/frames'
newdir = './images/wild/'
for file in os.listdir(dir):
    if i %20==0:
        os.rename(os.path.join(dir, file),os.path.join(newdir, file))
    i+=1