# Face_Detection_Recognition
> 老铁要是觉得可以，麻烦给个“Star”哦
## 说明
> 人脸检测和人脸识别的方法很多，项目《Face_Detection_Recognition》使用的是MTCNN人脸检测，FaceNet人脸识别 </br>
> 模型下载地址：百度网盘下载地址：链接: https://pan.baidu.com/s/1hAK9ylURkbeH52BtSSGWsw 提取码: jf1n </br>
> opencv-python的imread()函数并不支持中文路径，这里在提供一个函数read_image_gbk()方便读取中文路径的图像。</br>
## 文件说明
> create_dataset.py:用于生产人脸数据库</br>
> predict.py:人脸预测文件</br>
> batch_test.py:批量测试文件</br>
> evaluation_test.py:评价文件，用于绘制模型ROC曲线，测试文件只使用了agedb_30.bin人脸数据库</br>

## 参考资料
> 博客地址:《利用MTCNN和facenet实现人脸检测和人脸识别》https://panjinquan.blog.csdn.net/article/details/84896733</br>
> FaceNet的人脸识别效果并不算法好，目前测试使用InsightFace模型，在开数据集可以达到99.6%，在自建的数据集可以达到93%的准确率，比虹软的人脸识别率还高一点 </br>
> 关于InsightFace模型的项目，我还在整理，网友慢慢期待哈，不急！</br>
