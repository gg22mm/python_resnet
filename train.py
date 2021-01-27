# 导入各种python库
#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import random

# 获得数据集方法
# 最后是有关数据集的详细说明，我的数据集包括固体胶（danhong），和西瓜霜（huangtai）两种图片。
# 数据集链接：https://盘.拜读.com/s/1JOuGP412-rVFemsfzIHvjg 提取马：ybs8
def DataSet():
    
    # 首先需要定义训练集和测试集的路径，这里创建了 train ， 和 test 文件夹
    # 每个文件夹下又创建了 danhong，huangtai 两个文件夹，所以这里一共四个路径
    train_path_glue ='dataset/train/danhong/'
    train_path_medicine = 'dataset/train/huangtai/'
    train_path_danbai = 'dataset/train/danbai/'
    
    test_path_glue ='dataset/test/danhong/'
    test_path_medicine = 'dataset/test/huangtai/'
    test_path_danbai = 'dataset/train/danbai/'
    
    # os.listdir(path) 是 python 中的函数，它会列出 path 下的所有文件名
    # 比如说 imglist_train_glue 对象就包括了/train/danhong/ 路径下所有的图片文件名
    imglist_train_glue = os.listdir(train_path_glue)
    imglist_train_medicine = os.listdir(train_path_medicine)
    imglist_train_danbai = os.listdir(train_path_danbai)
    
    imglist_test_glue = os.listdir(test_path_glue)
    imglist_test_medicine = os.listdir(test_path_medicine)
    imglist_test_danbai = os.listdir(test_path_danbai)
    
    # 这里定义两个 numpy 对象，X_train 和 Y_train    
    # X_train 对象用来存放训练集的图片。每张图片都需要转换成 numpy 向量形式
    # X_train 的 shape 是 (360，224，224，3) 
    # 360 是训练集中图片的数量（训练集中固体胶和西瓜霜图片数量之和）
    # 因为 resnet 要求输入的图片尺寸是 (224,224) , 所以要设置成相同大小（也可以设置成其它大小，参看 keras 的文档）
    # 3 是图片的通道数（rgb）
    
    # Y_train 用来存放训练集中每张图片对应的标签
    # Y_train 的 shape 是 （360，2）
    # 360 是训练集中图片的数量（训练集中固体胶和西瓜霜图片数量之和）
    # 因为一共有两种图片，所以第二个维度设置为 2
    # Y_train 大概是这样的数据 [[0,1],[0,1],[1,0],[0,1],...]
    # [0,1] 就是一张图片的标签，这里设置 [1,0] 代表 固体胶，[0,1] 代表西瓜霜
    # 如果你有三类图片 Y_train 就因该设置为 (your_train_size,3)
    X_train = np.empty((len(imglist_train_glue) + len(imglist_train_medicine) + len(imglist_train_danbai), 224, 224, 3))  # 360，224，224，3  （360为图片总数,输入图片统一224,224，3为图片通道数）
    Y_train = np.empty((len(imglist_train_glue) + len(imglist_train_medicine) + len(imglist_train_danbai), 3))  # 360，2 （360为图片总数,2图片种类,因为一共有两种图片，所以第二个维度设置为 2） 更多类要改这里
    count = 0  # count 对象用来计数，每添加一张图片便加 1
    for img_name in imglist_train_glue:  # 遍历 /train/danhong 下所有图片，即训练集下所有的固体胶图片
        
        img_path = train_path_glue + img_name # 得到图片的路径+文件名
        img = image.load_img(img_path, target_size=(224, 224))   # 通过 image.load_img() 函数读取对应的图片,并转换成目标大小。 image 是 tensorflow.keras.preprocessing 中的一个对象
        img = image.img_to_array(img) / 255.0  # 将图片转换成 numpy 数组，并除以 255 ，归一化。 转换之后 img 的 shape 是 （224，224，3）
        
        X_train[count] = img  # 将处理好的图片装进定义好的 X_train 对象中
        
        #二分类：np.array((1,0))，np.array((0,1))
        #三分类：np.array((1,0,0))，np.array((0,1,0))，np.array((0,0,1))
        #四分类：np.array((1,0,0,0))，np.array((0,1,0,0))，np.array((0,0,1,0))，np.array((0,0,0,1))
        #更多类别数以此类推即可。
        #每种图片对应的标签顺序没有要求，自己使用模型预测的时候要清楚哪个标签对应哪类图片。
        Y_train[count] = np.array((1,0,0))   #[1,0] 代表 固体胶   -  将对应的标签装进 Y_train 对象中，这里都是 固体胶（danhong）图片，所以标签设为 [1,0]  更多类要改这里 
        count+=1
        
    for img_name in imglist_train_medicine:  # 遍历 /train/huangtai 下所有图片，即训练集下所有的西瓜霜图片

        img_path = train_path_medicine + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train[count] = np.array((0,1,0))  #[0,1] 代表西瓜霜 更多类要改这里
        count+=1

    for img_name in imglist_train_danbai:  # 遍历 /train/danbai 下所有图片，即训练集下所有的西瓜霜图片

        img_path = train_path_danbai + img_name        
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train[count] = np.array((0,0,1)) 
        count+=1
       
    # 下面的代码是准备测试集的数据，与上面的内容完全相同，这里不再赘述
    X_test = np.empty((len(imglist_test_glue) + len(imglist_test_medicine) + len(imglist_test_danbai), 224, 224, 3))
    Y_test = np.empty((len(imglist_test_glue) + len(imglist_test_medicine) + len(imglist_test_danbai), 3))  #更多类要改这里
    count = 0
    for img_name in imglist_test_glue:

        img_path = test_path_glue + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test[count] = np.array((1,0,0))   #更多类要改这里
        count+=1
        
    for img_name in imglist_test_medicine:
        
        img_path = test_path_medicine + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test[count] = np.array((0,1,0))  #更多类要改这里
        count+=1
    
    for img_name in imglist_test_danbai:
        
        img_path = test_path_danbai + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test[count] = np.array((0,0,1)) 
        count+=1

    # 打乱训练集中的数据 
    index = [i for i in range(len(X_train))]
    random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]
    
    # 打乱测试集中的数据
    index = [i for i in range(len(X_test))]
    random.shuffle(index)
    X_test = X_test[index]    
    Y_test = Y_test[index]

    return X_train,Y_train,X_test,Y_test


# 加载数据集
X_train,Y_train,X_test,Y_test = DataSet()
print('X_train shape : ',X_train.shape)
print('Y_train shape : ',Y_train.shape)
print('X_test shape : ',X_test.shape)
print('Y_test shape : ',Y_test.shape)


# 1、加载模型
# 然后载入 keras 中帮我们处理好的 resnet 模型
# 因为我的数据集较小，用的是 resnet50。载入模型一行代码就搞定。我的数据集中只有两种图片，设置参数 classes=2 。
# 这里的 classes 设置成你需要的类别数，有3个类别设为3，4个类别则设为4 更多类要改这里
# model = ResNet50(
#     weights=None,
#     classes=2
# )
model = keras.applications.ResNet50(
    weights=None,
    classes=3
)

# 这个方式报错了：module 'tensorflow._api.v2.train' has no attribute 'AdamOptimizer' （不必更改tensorflow 2）
# 2、接着在设置一些模型的参数，这里这里设置了优化器，loss计算方式，和 metrics。
# model.compile(optimizer=tf.train.AdamOptimizer(0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# 改成下面的
model.compile(optimizer = tf.optimizers.Adam(0.001),
              loss = 'categorical_crossentropy', #ategorical_crossentropy
              metrics=['accuracy'])




# train
# 3、训练模型 - 完成了上面两步之后就可以开始训练了（数据集的处理在后面），也是一行代码搞定：
model.fit(X_train, Y_train, epochs=1, batch_size=6)

# evaluate
# 评估模型 - 模型训练好之后，可以在测试集上评估一下模型的性能：
model.evaluate(X_test, Y_test, batch_size=32)

# save
# 保存模型 - 如果经过模型评估和测试都没有问题的话可以把模型保存起来，方便后面的使用
model.save('my_resnet_model.h5')

# test
# 测试 - 最后还要用单张图片测试一下效果，这里只要修改图片的路径就可以测试不同的图片了：
img_path = "dataset/test/danhong/1_danhongshe1.jpg"
# img_path = "dataset/test/huangtai/2_huangtai1.jpg"
img = image.load_img(img_path, target_size=(224, 224))

# plt.imshow(img)
img = image.img_to_array(img)/ 255.0
img = np.expand_dims(img, axis=0)  # 为batch添加第四维

print(model.predict(img))
np.argmax(model.predict(img))
