# 这个现在不用报，都集成到rand_images.py中了
import os
import cv2

# 裁剪图片
def bullImages(image_path,distinct_path):
    count=0
    if os.path.isdir(image_path):
        for img_name in os.listdir(image_path):
            img_path = image_path + img_name
            count+=1
            print(img_path)
            # exit()        
            img = cv2.imread(img_path, 1)  # img_name为字符串，要读取的图片的名字或者全路径；command可以为空，默认值为1；0  表示以灰度图方式读取图片；1  表示以彩色度方式读取图片
            # cv2.imshow('image', img)  # 创建显示框并显示图片
            # namedWindow(name, 1)  # 只创建可调节显示框，需与imshow配合使用
            #size = img.shape  # 获取图片大小（长，宽，通道数）
            tempimg = cv2.resize(img, (224,224), cv2.INTER_AREA)#缩放图片推荐选择INTER_AREA
            # gray = cv2.cvtColor(tempimg, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('imag2', tempimg)
            # 等待用户关闭显示框
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            cv2.imwrite(distinct_path+img_name, tempimg)


# 根目录
dirPath='/usr/local/servers/web/www/txsb/koutu/resnet/3/bull_images/1/'
  
for root, dirs, files in os.walk(dirPath):  
    # print(root) #当前目录路径  
    # print(dirs) #当前路径下所有子目录  
    # print(files) #当前路径下所有非目录子文件

    if dirs :
        print(dirs)

        for dir in dirs :
         
            #循环根目录下的所有目录
            image_path=dirPath+dir+'/'
            print(image_path)
            distinct_path=image_path #"/usr/local/servers/web/www/txsb/koutu/resnet/3/bull_images/new/danbai"
            bullImages(image_path,distinct_path)

