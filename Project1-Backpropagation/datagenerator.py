import numpy as np
import random
from PIL import Image, ImageDraw

# Not commented !!

class datagenerator:

    def __init__(self, n, size, noise, centered=True):
        self.n = n  #size of grid
        self.size = size    #size of dataset
        self.noise = noise  #need this !!
        self.centered = centered


    def generate(self):
        size = self.size
        n = self.n
        images = []
        labels = np.zeros((size, 4))
        for i in range(size):
            im = Image.new("1", (n, n), color=1)
            #draw = ImageDraw.Draw(im)
            method = random.randint(1,4)
            if method == 1:
                self.rectangle(im, n)
            if method == 2:
                self.circle(im, n)
            if method == 3:
                self.lines(im, n)
            if method == 4:
                self.cross(im, n)

            #functionality for adding noise

            images.append(im)
            labels[i][method-1] = 1
            # im.show()                             # <---- show images
        return images, labels
            

    def rectangle(self, image, n):
        draw = ImageDraw.Draw(image)
        if self.centered:
            x0 = random.randint(0,n//2-2)
            y0 = random.randint(0,n//2-2)
            x1 = n-x0-1
            y1 = n-y0-1
        else:
            x = random.randint(3,n-1)
            y = random.randint(3,n-1)
            x0 = random.randint(0, n-x-1)
            y0 = random.randint(0, n-y-1)
            x1 = x0+x
            y1 = y0+y

        draw.rectangle((x0, y0, x1, y1), outline=0)


    def circle(self, image, n):
        draw = ImageDraw.Draw(image) 
        if self.centered:
            x0 = random.randint(0,n//2-3)
            y0 = x0
            x1 = n-x0-1
            y1 = x1
        else:
            diameter = random.randint(3,n-1)
            x0 = random.randint(0, n-diameter-1)
            y0 = random.randint(0, n-diameter-1)
            x1 = x0+diameter
            y1 = y0+diameter

        draw.ellipse((x0, y0, x1, y1), outline=0)


    def lines(self, image, n):
        #horisontal lines
        draw = ImageDraw.Draw(image)
        #number of lines
        lines = random.randint(2,n//3)
        #position for the lines
        pos = random.sample(range(n), lines)
        for p in pos:
            draw.line((0, p, n, p), width=1)


    def cross(self, image, n):
        draw = ImageDraw.Draw(image)
        if self.centered:
            x0 = random.randint(0,n//2-3)
            y0 = random.randint(0,n//2-3)
            x1 = n-x0-1
            y1 = n-y0-1
        else:
            x = random.randint(3,n-1)
            y = random.randint(3,n-1)
            x0 = random.randint(0, n-x)
            y0 = random.randint(0, n-y)
            x1 = x0+x
            y1 = y0+y

        width = random.randint(1, min((y1-y0)//2, (x1-x0)//2))
        #draw horisontal line
        draw.line((x0, (y0+y1)//2, x1, (y0+y1)//2), width=width)
        #draw vertical line
        draw.line(((x0+x1)//2, y0, (x0+x1)//2, y1), width=width)
        
    def im2grid(self, images):
        # images: list of images
        # returns: array of grids representing all images
        gridarray = np.zeros((self.size, self.n, self.n))
        for im in range(self.size):
           gridarray[im] = np.asarray(images[im]).astype(int)
        return gridarray
    
    def grid2array(self, gridarray):
        # gridarray: array of grids
        # returns: array of arrays representing all images
        arr = np.zeros((self.size, self.n * self.n))
        for i in range(self.size):
            arr[i] = gridarray[i].flatten()
        return arr



# --------------------------- test -------------------------------

gen = datagenerator(50, 7, 0, centered=False)

images, target = gen.generate()
#print(images)
#print(np.array(images[0]).astype(int))
mat = gen.im2grid(images)
#print(mat)
#print(mat[0].flatten())
#print(gen.grid2array(mat))
print(target)
