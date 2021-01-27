# 导入各种python库
#!/usr/bin/env python
# coding: utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# restore
# 恢复模型 - 使用已经成功保存的模型也很简单：
model = tf.keras.models.load_model('my_resnet_model.h5')


# test
# 测试 - 最后还要用单张图片测试一下效果，这里只要修改图片的路径就可以测试不同的图片了：
# img_path = "dataset/test/danhong/1_danhongshe1.jpg" 
# img_path = "dataset/test/huangtai/2_huangtai1.jpg"          #[[0.9989537  0.00104625]]
# img_path = "dataset/test/danbai/2_danbaishe12.jpg"          #[[0.9989537  0.00104625]]
img_path = "dataset/test/danbai/baitai3.jpg"          
img = image.load_img(img_path, target_size=(224, 224))

plt.imshow(img)
img = image.img_to_array(img)/ 255.0
img = np.expand_dims(img, axis=0)  # 为batch添加第四维

# 测试结果如下：0 代表固体胶， 1 代表西瓜霜   | 固体胶（glue），和西瓜霜（medicine）两种图片。 
print(model.predict(img))
np.argmax(model.predict(img))
