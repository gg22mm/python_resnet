# 图片数据不够快来试试使用imgaug增强数据
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import os

#设置随机数种子
rundNumber=5
ia.seed(rundNumber)

# 方法一、生成随机图片
# rundImages('E:/wll/phpsys/www/dzkj_zcy/zcy/txsb/koutu/resnet/3/bull_images/example.jpg','example.jpg','/usr/local/servers/web/www/txsb/koutu/resnet/3/bull_images/1/')
def rundImages(picPath,imgName,dirPath):
    
    #读取图片
    example_img = cv2.imread(picPath)

    #通道转换,开这个的话，显示正常，但是写文件就不正常了
    # example_img = example_img[:, :, ::-1]
    
    #对图片进行缩放处理
    example_img = cv2.resize(example_img,(224,224))

    seq = iaa.Sequential([        

        #随机裁剪图片边长比例的0~0.1
        iaa.Crop(percent=(0,0.1),keep_size=True),

        #从图片边随机裁剪50~100个像素,裁剪后图片的尺寸和之前不一致
        #通过设置keep_size为True可以保证裁剪后的图片和之前的一致
        # iaa.Crop(px=(50,100),keep_size=True),

        #50%的概率水平翻转
        iaa.Fliplr(0.5),

        #50%的概率垂直翻转
        iaa.Flipud(0.5),

        #Sometimes是指指针对50%的图片做处理
        iaa.Sometimes(
            0.5,
            #高斯模糊
            iaa.GaussianBlur(sigma=(0,0.5))
        ),

        #增强或减弱图片的对比度
        iaa.LinearContrast((0.75,1.5)),

        #添加高斯噪声
        #对于50%的图片,这个噪采样对于每个像素点指整张图片采用同一个值
        #剩下的50%的图片，对于通道进行采样(一张图片会有多个值)
        #改变像素点的颜色(不仅仅是亮度)
        iaa.AdditiveGaussianNoise(loc=0,scale=(0.0,0.05*255),per_channel=0.5),
        #让一些图片变的更亮,一些图片变得更暗
        #对20%的图片,针对通道进行处理
        #剩下的图片,针对图片进行处理
        iaa.Multiply((0.8,1.2),per_channel=0.2),
        #仿射变换
        iaa.Affine(
            #缩放变换
            scale={"x":(0.8,1.2),"y":(0.8,1.2)},
            #平移变换
            translate_percent={"x":(-0.2,0.2),"y":(-0.2,0.2)},
            #旋转
            rotate=(-25,25),
            #剪切
            shear=(-8,8)
        )
    #使用随机组合上面的数据增强来处理图片
    ],random_order=True)
   
    #生成一个图片列表    
    i=0
    while i<rundNumber:
        i=i+1
        aug_example_img = seq.augment_image(image=example_img)
        # ia.imshow(aug_example_img)    
        cv2.imwrite(dirPath+str(i)+'_'+imgName, aug_example_img)    



# 方法二、获取图片全地址
def getImagesAllPath(image_path,distinct_path):   
    if os.path.isdir(image_path):
        for img_name in os.listdir(image_path):
            img_path = image_path + img_name            
            print(img_path)
            # print(img_name)
            # 随机生成图片
            rundImages(img_path,img_name,distinct_path)
            

# 根目录
dirPath='E:/wll/phpsys/www/dzkj_zcy/zcy/txsb/koutu/resnet/3/bull_images/1/'
  
for root, dirs, files in os.walk(dirPath):  
    # print(root) #当前目录路径  
    # print(dirs) #当前路径下所有子目录  
    # print(files) #当前路径下所有非目录子文件

    if dirs :
        print(dirs)

        for dir in dirs :
         
            # 循环根目录下的所有目录
            image_path=dirPath+dir+'/'
            print(image_path)
            distinct_path=image_path #"/usr/local/servers/web/www/txsb/koutu/resnet/3/bull_images/new/danbai"
            getImagesAllPath(image_path,distinct_path)