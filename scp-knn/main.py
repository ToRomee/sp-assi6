import matplotlib.pyplot as plt
import random
import numpy as np
from setup import setup
from Triangle import isInside  
from sklearn.neighbors import KNeighborsClassifier


n = 500
k = 5 
f = 0

#打印生成数据
x_vals, y_vals, classifs = setup(n, k, f)
print(x_vals[:5])
print(classifs[:5])

perimeter_values = [7, 10, 13, 16, 19, 22, 25, 28]
misclassified_counts = []

#坐标转换函数
def scale_triangle(coords, side):
  x1, y1, x2, y2, x3, y3 = coords  
  scaled_x1 = x1 / side
  scaled_y1 = y1 / side
  scaled_x2 = x2 / side 
  scaled_y2 = y2 / side
  scaled_x3 = x3 / side
  scaled_y3 = y3 / side  

  return [scaled_x1, scaled_y1, scaled_x2, scaled_y2, scaled_x3, scaled_y3]


#计算误差函数
def count_errors(coords, knn, x_vals, y_vals):

  print(type(coords))

  scaled_coords = np.array(coords)

  scaled_coords = scaled_coords.reshape(-1,1)  

  print(scaled_coords.shape, x_vals.shape)   

  predictions = knn.predict(scaled_coords)

  print(predictions)

  correct_label = 0 if random.random() < 0.5 else 1

  if predictions[0] != correct_label:
    print("Prediction wrong!")
    return 1
  else:  
    return 0

for perimeter in perimeter_values:

  side = perimeter / 3.414213562  

  #打印三角形值    
  print(side)
  
  x1, y1 = 0, 0
  x2, y2 = side, 0  
  x3, y3 = side, side

  #打印转换函数
  scaled_coords = scale_triangle([x1, y1, x2, y2, x3, y3], side)
  print(scaled_coords)

  knn = KNeighborsClassifier(n_neighbors=k)

  #打印训练结果
  knn.fit(x_vals, classifs)
  print('KNN fitted')

  misclassified_count = count_errors(scaled_coords, knn, x_vals, y_vals)   
   
  #打印误差结果
  print(misclassified_count)

  misclassified_counts.append(misclassified_count)

  print(f"Perimeter {perimeter}: {misclassified_count} misclassifications")
  
# 打印结果  
print(misclassified_counts)

plt.bar(perimeter_values, misclassified_counts)
plt.savefig('result.png') 
print('Done')