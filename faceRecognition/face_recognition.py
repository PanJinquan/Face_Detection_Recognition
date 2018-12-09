# -*-coding: utf-8 -*-
"""
    @Project: faceRecognition
    @File   : face_recognition.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-12-07 11:33:30
"""
import facenet
import tensorflow as tf
import align.detect_face as detect_face

class facenetEmbedding:
    def __init__(self,model_path):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        # Load the model
        facenet.load_model(model_path)
        # Get input and output tensors
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.tf_embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    def  get_embedding(self,images):
        feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
        embedding = self.sess.run(self.tf_embeddings, feed_dict=feed_dict)
        return embedding
    def free(self):
        self.sess.close()


class Facedetection:
    def __init__(self):
        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor
        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            # gpu_memory_fraction = 1.0
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            sess = tf.Session()
            with sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)
    def detect_face(self,image):
        bounding_boxes, points = detect_face.detect_face(image, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        return  bounding_boxes, points


def detection_face(img):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        # gpu_memory_fraction = 1.0
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    return bounding_boxes,points