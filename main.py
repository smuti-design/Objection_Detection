import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imageai.Detection import ObjectDetection
import os
import tensorflow as tf

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.1.0.h5"))
detector.loadModel()
detections = detector.detectObjectFromImage(input_image=os.path.join(execution_path, "image.jpg"),
                                            output_image_path=os.path.join(execution_path,"image_new.jpg"))
for eachObject in detections:
    print(eachObject["Name"], ":" ,eachObject["percentage_probability"])