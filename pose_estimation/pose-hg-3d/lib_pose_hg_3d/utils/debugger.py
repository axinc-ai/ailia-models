import numpy as np
import cv2
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
  
def show_2d(img, points, c, edges):
  num_joints = points.shape[0]
  points = ((points.reshape(num_joints, -1))).astype(np.int32)
  for j in range(num_joints):
    cv2.circle(img, (points[j, 0], points[j, 1]), 3, c, -1)
  for e in edges:
    if points[e].min() > 0:
      cv2.line(img, (points[e[0], 0], points[e[0], 1]),
                    (points[e[1], 0], points[e[1], 1]), c, 2)
  return img

mpii_edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], 
              [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], 
              [6, 8], [8, 9]]

class Debugger(object):
  def __init__(self, ipynb=False, edges=mpii_edges):
    self.ipynb = ipynb
    if not self.ipynb:
      self.plt = plt
      self.fig = self.plt.figure()
      self.ax = self.fig.add_subplot((111),projection='3d')
      self.ax.grid(False)
    oo = 1e10
    self.xmax, self.ymax, self.zmax = -oo, -oo, -oo
    self.xmin, self.ymin, self.zmin = oo, oo, oo
    self.imgs = {}
    self.edges=edges
    

  
  def add_point_3d(self, points, c='b', marker='o', edges=None):
    if edges == None:
      edges = self.edges
    #show3D(self.ax, point, c, marker = marker, edges)
    points = points.reshape(-1, 3)
    x, y, z = np.zeros((3, points.shape[0]))
    for j in range(points.shape[0]):
      x[j] = points[j, 0].copy()
      y[j] = points[j, 2].copy()
      z[j] = - points[j, 1].copy()
      self.xmax = max(x[j], self.xmax)
      self.ymax = max(y[j], self.ymax)
      self.zmax = max(z[j], self.zmax)
      self.xmin = min(x[j], self.xmin)
      self.ymin = min(y[j], self.ymin)
      self.zmin = min(z[j], self.zmin)
    if c == 'auto':
      c = [(z[j] + 0.5, y[j] + 0.5, x[j] + 0.5) for j in range(points.shape[0])]
    self.ax.scatter(x, y, z, s = 200, c = c, marker = marker)
    for e in edges:
      self.ax.plot(x[e], y[e], z[e], c = c)
    
  def show_3d(self):
    max_range = np.array([self.xmax-self.xmin, self.ymax-self.ymin, self.zmax-self.zmin]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(self.xmax+self.xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(self.ymax+self.ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(self.zmax+self.zmin)
    for xb, yb, zb in zip(Xb, Yb, Zb):
      self.ax.plot([xb], [yb], [zb], 'w')
    self.plt.show()


    
  def add_img(self, img, imgId = 'default'):
    self.imgs[imgId] = img.copy()
  
  def add_mask(self, mask, bg, imgId = 'default', trans = 0.8):
    self.imgs[imgId] = (mask.reshape(mask.shape[0], mask.shape[1], 1) * 255 * trans + \
                        bg * (1 - trans)).astype(np.uint8)

  def add_point_2d(self, point, c, imgId='default'):
    self.imgs[imgId] = show_2d(self.imgs[imgId], point, c, self.edges)
  
  def show_img(self, pause = False, imgId = 'default'):
    cv2.imshow('{}'.format(imgId), self.imgs[imgId])
    if pause:
      cv2.waitKey()
  
  def show_all_imgs(self, pause = False):
    if not self.ipynb:
      for i, v in self.imgs.items():
        cv2.imshow('{}'.format(i), v)
      if pause:
        cv2.waitKey()
    else:
      self.ax = None
      nImgs = len(self.imgs)
      fig=plt.figure(figsize=(nImgs * 10,10))
      nCols = nImgs
      nRows = nImgs // nCols
      for i, (k, v) in enumerate(self.imgs.items()):
        fig.add_subplot(1, nImgs, i + 1)
        if len(v.shape) == 3:
          plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
        else:
          plt.imshow(v)
  
  def save_3d(self, path):
    max_range = np.array([self.xmax-self.xmin, self.ymax-self.ymin, self.zmax-self.zmin]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(self.xmax+self.xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(self.ymax+self.ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(self.zmax+self.zmin)
    for xb, yb, zb in zip(Xb, Yb, Zb):
      self.ax.plot([xb], [yb], [zb], 'w')
    self.plt.savefig(path, bbox_inches='tight', frameon = False)
  
  def save_img(self, imgId = 'default', path = '../debug/'):
    cv2.imwrite(path + '{}.png'.format(imgId), self.imgs[imgId])
    
  def save_all_imgs(self, path = '../debug/'):
    for i, v in self.imgs.items():
      cv2.imwrite(path.format(i), v)
    
