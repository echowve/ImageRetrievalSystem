import os
import cv2
import numpy as np
from collections import OrderedDict
from PyQt5.QtWidgets import QLabel as QLabel_origin
from PyQt5.QtCore import pyqtSignal
from PyQt5 import QtCore, QtGui, QtWidgets


#配置数据集的类
class LogoDataset():
    # 创建数据集字典，{索引：文件名}
    def __init__(self, dir_):
        self.dir = dir_
    
    def getdataset(self):
        imgs  = []
        for root, _, files in os.walk(self.dir):
            for f in files:
                if f.endswith('jpg'):
                    imgs.append(os.path.join(root,f))

        self.genFile(imgs)
        return OrderedDict(zip(range(len(imgs)), imgs))

    def genFile(self, imgs):
        with open('data\\url.txt', 'w') as f:
            for i in imgs:
                f.write(i)
                f.write(' ')
                f.write('https://www.baidu.com\n')





if __name__=='__main__':
    dataset_dir = 'E:\\Project\\LogoRetrieval\\logo'
    dataset = LogoDataset(dataset_dir).getdataset()
    img = cv2.imread(dataset[0])
    print(len(dataset))
else:
    from utils.config import dataset_dir


def sortlabels_gui(labels):
    #对gui label（图片框）排序的函数
    return labels.sort(key=keyfun)

#保存数据的类，用于数据管理
class Class_DATA():
    pass

#比较函数，用于排序
def keyfun(x):
    r1 = x.split('_')
    if len(r1)<2:
        return 0
    elif 'query' in r1 or 'gnd' in r1:
        return 1000000
    else:
        return int(r1[1])

#用于返回指定图片在数据集images中的字典键，
def getIndex(image, images):

    image = image.replace('/', '\\')
    for index, img in images.items():
        if img==image:
            return index

def EuclideanDist(probe_feat, gallery_feats):
#欧式距离计算，sqrt((a - b)^2)
    sub = probe_feat - gallery_feats
    powered = np.power(sub, 2)
    dist = np.sqrt(np.sum(powered, axis=1))
    return dist

#重新定义的Qlabel类，用于给label增加clicked信号
class QLabel(QLabel_origin):
    clicked = pyqtSignal(QLabel_origin)
    def __init__(self, para):
        super().__init__(para)

    def mouseReleaseEvent(self, QMouseEvent):
        if QMouseEvent.button() == QtCore.Qt.LeftButton:
            self.clicked.emit(self)


#从配置文件读取url的类
def getUrl(url_file_dir):

    result = {}
    with open(url_file_dir) as f:
        l  = f.readline()
        while l:
            file_name, url = l.split(' ')
            t = {file_name:url}
            result.update(t)
            l = f.readline()
    return result


