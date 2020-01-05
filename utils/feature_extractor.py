import numpy as np
import cv2
from model.resnet import ResNetFeat
from model.color import Color
from tqdm import tqdm

class FeatureEx():
    def __init__(self, name='resnet34'):

        feat_index = [0, 1]
        self.deepextractor = None
        self.extractor = None

        if name.find('resnet') != -1:
            feat_index = [1, 0]
        if name=='fusion':
            feat_index = [1, 1]

        self.name = name
        if feat_index[0]:
            self.deepextractor = ResNetFeat(model_name='resnet34', use_gpu=True)
            #深度特征提取器
        if feat_index[1]:
            self.extractor = Color()
        #传统特征提取器


    def __call__(self, images):
        f = list()
        print('begin extract features')
        for img_path in tqdm(images):
            vector1 = []
            vector2 = []
            if self.deepextractor:
                img = cv2.imread(images[img_path])
                vector1 = np.squeeze(self.deepextractor.getfeature(img)).tolist()
            if self.extractor:
                img = cv2.imread(images[img_path])
                vector2 = self.extractor.histogram(img)
                vector2 = np.squeeze(vector2)
                vector2 = vector2.tolist()
            vector1.extend(vector2) #拼接两种特征
            f.append(vector1)
        r = np.array(f)
        return r
            

        

