from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random
import math
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt

def isInside(x, y, vtx):
  x1, y1 = vtx[0]
  x2, y2 = vtx[1]
  x3, y3 = vtx[2]

  tri_base = math.sqrt((x2-x1)**2 + (y2-y1)**2) 
  tri_height = 2 * abs((x3*(y2-y1) + x1*(y3-y2) + x2*(y1-y3)) / tri_base)

  tri_area = tri_base * tri_height / 2

  area = abs((x*(y2-y3) + x2*(y3-y) + x3*(y-y1) - y*(x2-x3) - y2*(x3-x) - y3*(x-x1))/2.0)

  return area < tri_area/2

def generate_points(n, f=0, vtx3=(0,0,0)):
  S = np.zeros((n,2))
  color = np.zeros(n)

  for i in range(n):
    x = random.uniform(0, 10)
    y = random.uniform(0, 10) 

    if isInside(x, y, vtx3[0],vtx3[1]):
      color[i] = 1

  return S, color

def run_experiment(n, k=5, perimeters=[7,10,13,16,19,22,25,28]):

  scores = []

  for perimeter in perimeters:

    side = perimeter / 3.414213562

    vtx3 = (side, side, side)
    
    for i in range(20):

        S, color = generate_points(n, f=0, vtx3=vtx3)
        test_points, test_color = generate_points(10000, f=0, vtx3=vtx3)
          
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(S, color)
        
        pred_color = knn.predict(test_points)
        score = accuracy_score(test_color, pred_color)

        scores.append(score)

    avg_score = np.mean(scores)
    std_score = np.std(scores)

    print(f"Perimeter {perimeter}: {avg_score}, {std_score}")

  return scores

scores = run_experiment(500)

perimeters = [7,10,13,16,19,22,25,28]
plt.bar(perimeters, scores) 
plt.savefig('result.png')