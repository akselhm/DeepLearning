import numpy as np
import random
from PIL import Image, ImageDraw


class datagenerator:

    def __init__(self, n, size, noise, centered='True'):
        self.n = n  #size of grid
        self.size = size    #size of dataset
        self.noise = noise
        self.centered = centered

    def generate(self):
        size = self.size
        n = self.n
        images = []
        labels = np.zeros(size)
        for i in range(size):
            im = Image.new(1, (n, n), color=1)
            #draw = ImageDraw.Draw(im)

            

    def rectangle(self, image, n):
        draw = ImageDraw.Draw(image)
        if self.centered:
            x0 = random.randint(0,n//2-1)
            y0 = random.randint(0,n//2-1)
            x1 = n-x0-1
            y1 = n-y0-1
        else:
            x0 = random.randint(0,n-3)
            y0 = random.randint(0,n-3)
            x0 = random.randint(x0+2,n-1)
            y0 = random.randint(y0+2,n-1)

        draw.rectangle((x0, y0, x1, y1), outline=0)
        return 

    """
    def generate(self):
        size = self.size
        n = self.n
        grid = np.zeros((size, n, n))
        target = np.zeros(size)
        for i in range(size):
            method = random.randint(1,2)
            # create grid (image)
            if method == 1:
                grid[i] = self.vlines(grid[i], n)
            if method == 2:
                grid[i] = self.hlines(grid[i], n)
            if method == 3:
                grid[i] = self.square(grid[i], n)
            #insert noise to image

            # add target class
            target[i] = method
        
        return grid, target

    
    def vlines(self, grid, n):
        #number of lines
        lines = random.randint(2,n//3)
        #position for the lines
        pos = random.sample(range(n), lines)
        for p in pos:
            grid[:][p] = 1
        return grid

    def hlines(self, grid, n):
        #number of lines
        lines = random.randint(2,n//3)
        #position for the lines
        pos = random.sample(range(n), lines)
        for p in pos:
            grid[p][:] = 1
        return grid

    def square(self, grid, n):
        #TODO: make function
        return grid

gen = datagenerator(20, 3, 0)

images = gen.generate()

for i in range(3):
    print image(images[i])
"""