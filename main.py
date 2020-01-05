from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap
from tools.tools import sortlabels_gui, Class_DATA, LogoDataset, EuclideanDist, getUrl
from utils.feature_extractor import FeatureEx
from UI.logo import Ui_MainWindow
from utils.config import dataset_dir, probe_dir, feat_type, url_file_dir
import numpy as np
import webbrowser as web
import sys, os

class UI_Main(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(UI_Main, self).__init__()
        self.setGeometry(300, 300, 500, 300)
        self.setWindowTitle('System')
        self.setupUi(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("retrieval.ico"),QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.labels = [i for i in self.__dict__ if i.startswith('label')]

        #为每个图片显示框设置槽函数用于点击图片打开网页
        for label in self.labels:
            label = eval('self.' + label)
            label.clicked.connect(self.OpenUrl) 

        sortlabels_gui(self.labels)
        #加载按钮和查询按钮的槽函数
        self.pushButton_select.clicked.connect(self.load)
        self.pushButton_query.clicked.connect(self.query)

        #特征提取器
        self.feaex = FeatureEx(name=feat_type)
        
        #初始化在库集，整体数据集，用字典存储，同时提取待查数据集的特征
        self.init_dataset()

        #在查询前做一些初始化，包括数据设置界面，重置之前查询的数据
        self.InitEachQuery()

        #显示界面
        self.show()

    def InitEachQuery(self):
        self.data = Class_DATA() #重新初始化，青空之前数据self.data保存的数据

        #设置图片显示框格式
        for label in self.labels:
            l = eval('self.' + label)
            l.clear()
            l.setStyleSheet("font:15pt '楷体';border-width: 2px;border-style: solid;border-color: rgb(50, 50, 50);")
            l.setScaledContents(True)


    def init_dataset(self):

        self.data = Class_DATA()
        dataset = LogoDataset(dataset_dir).getdataset()
        self.gallery_images = dataset
        self.gallery_feature = self.feaex(dataset)
        self.url = getUrl(url_file_dir)


    def showImages(self):
        #根据排序的索引显示图像
        images = self.gallery_images #之前提取的数据库
        ranklist = self.data.ranklist #索引
        cur_index_begin = 0
        cur_index_end = 20
        for i, index in enumerate(ranklist[cur_index_begin:cur_index_end]):
            pixmap = QPixmap(images[index])
            label = eval('self.' + self.labels[i])
            label.Picmap = pixmap
            rank = cur_index_begin + i + 1
            label.rank = rank
            label.index = index
            img_name = os.path.split(images[index])[-1]
            label.setToolTip(img_name[:-4] + 'rank@ ' + str(rank)) #设置tip，鼠标在图片停留时显示
            label.setPixmap(pixmap)
            label.url = self.url[images[index]] #保存链接，用于点击图片时调用浏览器


    def load(self):

        fname, success = QFileDialog.getOpenFileName(self, 'Select Query Image', probe_dir,
                                                     'Image files(*.jpg *.gif *.png *bmp)')
        if success:
            self.InitEachQuery()
            self.data.query_image_name = fname
            self.data.probe_feature = self.feaex({0:fname}) #提取查询特征
            self.label_query.setPixmap(QPixmap(fname))

    def query(self):

        #计算欧式距离，查询
        self.data.distance = EuclideanDist(self.data.probe_feature, self.gallery_feature)
        #距离从小到大的索引
        self.data.ranklist = np.argsort(self.data.distance)
        #显示查询结果
        self.showImages()

    def OpenUrl(self, label):
        #打开浏览器访问设置的链接
        web.open_new(label.url)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = UI_Main()
    sys.exit(app.exec_())