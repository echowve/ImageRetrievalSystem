#待查数据集
dataset_dir = 'logo'
#查询图片保存的位置
probe_dir = 'logo'
#特征选择，1 传统颜色直方图特征还是 2 深度特征还是 3 传统和深度特征拼接的融合特征
feat_type = 'fusion' # resnet34 color fusion

#点击图片时通过浏览器访问链接的配置文件，具体看文件内容
url_file_dir = 'data\\url.txt'