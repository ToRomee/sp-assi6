import random
import numpy as np
from matplotlib import pyplot as plt
from Triangle import isInside

# plot the starting 500 values with triangle separator
def plot_stuff(x_vals, y_vals, classifs):
    blue_x = [x_vals[i] for i in range(len(x_vals)) if classifs[i] == 1]
    blue_y = [y_vals[i] for i in range(len(y_vals)) if classifs[i] == 1]
    red_x = [x_vals[i] for i in range(len(x_vals)) if classifs[i] == 0]
    red_y = [y_vals[i] for i in range(len(y_vals)) if classifs[i] == 0]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect('equal', adjustable='box')

    plt.scatter(blue_x,blue_y, s=10, c="blue")
    plt.scatter(red_x,red_y, s=10, c="red")

    plt.grid()
    plt.show()

def setup(n, k, f):

  x_vals = []
  y_vals = []
  classifs = []

  for i in range(0, n):
      x = random.uniform(0, 10)  
      y = random.uniform(0, 10)
      classif = 0

      if (not isInside(3,3,7,3,7,7, x, y)):
          classif = 1

      # flip classfication by f        
      if (random.uniform(0, 1) < f):
          classif = 1 - classif
          
      x_vals.append(x)
      y_vals.append(y)
      classifs.append(classif)
      
  x_vals = np.array(x_vals).reshape(-1,1) 
  y_vals = np.array(y_vals)

  plot_stuff(x_vals, y_vals, classifs)

  return x_vals, y_vals, classifs
