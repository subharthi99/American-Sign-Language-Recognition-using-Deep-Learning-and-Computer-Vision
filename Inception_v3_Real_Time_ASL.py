# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import cv2
import gc

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

def convert_label_to_class(label):
    temp = int(label)
    if(label == 0):
        return 'A'
    if(label == 1):
        return 'B'
    if(label == 2):
        return 'C'
    if(label == 3):
        return 'D'
    if(label == 4):
        return 'E'
    if(label == 5):
        return 'F'
    if(label == 6):
        return 'G'
    if(label == 7):
        return 'H'
    if(label == 8):
        return 'I'
    if(label == 9):
        return 'J'
    if(label == 10):
        return 'K'
    if(label == 11):
        return 'L'
    if(label == 12):
        return 'M'
    if(label == 13):
        return 'N'
    if(label == 14):
        return 'O'
    if(label == 15):
        return 'P'
    if(label == 16):
        return 'Q'
    if(label == 17):
        return 'R'
    if(label == 18):
        return 'S'
    if(label == 19):
        return 'T'
    if(label == 20):
        return 'U'
    if(label == 21):
        return 'V'
    if(label == 22):
        return 'W'
    if(label == 23):
        return 'X'
    if(label == 24):
        return 'Y'
    if(label == 25):
        return 'Z'
    if(label == 26):
        return 'Delete'
    if(label == 27):
        return 'Nothing'
    if(label == 28):
        return 'Space'

feed_transformation = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((75,75)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

class Inception_Comp1(nn.Module):
    def __init__(self,input_channels, extract_features):
        super(Inception_Comp1, self).__init__()
        self.leaf1_3_3 = conv_2D_overwrite(input_channels, 8, kernel_size = 1)
        self.leaf2_3_3 = conv_2D_overwrite(8, 16, kernel_size = 3, padding = 1)
        self.leaf3_3_3 = conv_2D_overwrite(16, 32, kernel_size = 3, padding = 1)
        
        self.leaf1_5_5 = conv_2D_overwrite(input_channels, 8, kernel_size = 1)
        self.leaf2_5_5 = conv_2D_overwrite(8, 16, kernel_size = 5, padding = 2)
        
        self.leaf_1_1 = conv_2D_overwrite(input_channels, 8, kernel_size = 1)
        
        self.leaf4 = conv_2D_overwrite(input_channels, extract_features, kernel_size = 1)
        
    def forward(self, x):
        #print('*')
        leaf_1 = self.leaf_1_1(x)
        #print(leaf_1.shape)
        #print('**')
        leaf_5 = self.leaf1_5_5(x)
        #print(leaf_5.shape)
        #print('***')
        leaf_5 = self.leaf2_5_5(leaf_5)
        #print(leaf_5.shape)
        #print(x.shape)
        #print('****')
        leaf_3 = self.leaf1_3_3(x)
        #print(leaf_3.shape)
        #print('*****')
        leaf_3 = self.leaf2_3_3(leaf_3)
        #print(leaf_3.shape)
        #print('******')
        leaf_3 = self.leaf3_3_3(leaf_3)
        #print(leaf_3.shape)
        #print('*******')
        extraction_leaf = F.avg_pool2d(x, kernel_size = 3, stride = 1, padding = 1)
        extraction_leaf = self.leaf4(extraction_leaf)
        #print('********')
        #leaf_1 = self.leaf_1_1(x)
        
       
        combinations = []
        
        combinations.append(leaf_1)
        combinations.append(leaf_5)
        combinations.append(leaf_3)
        combinations.append(extraction_leaf)
        #print('*******')
        concatenation = torch.cat(combinations, 1)
        #print('********')
        return concatenation

class Inception_Comp2(nn.Module):
    def __init__(self,input_channels):
        super(Inception_Comp2, self).__init__()
        self.leaf1_3_3 = conv_2D_overwrite(input_channels, 32, kernel_size = 1)
        self.leaf2_3_3 = conv_2D_overwrite(32, 64, kernel_size = 3, padding = 1)
        self.leaf3_3_3 = conv_2D_overwrite(64, 96, kernel_size = 3, stride = 2)
        
        self.leaf3 = conv_2D_overwrite(input_channels, 32, kernel_size = 3, stride = 2)
        
    def forward(self, x):
        
        leaf_extract = F.max_pool2d(x, kernel_size = 3, stride = 2)
        
        leaf_1 = self.leaf3(x) 
        
        leaf_3 = self.leaf1_3_3(x)
        leaf_3 = self.leaf2_3_3(leaf_3)
        leaf_3 = self.leaf3_3_3(leaf_3)
        
        combinations = []
        
        combinations.append(leaf_1)
        combinations.append(leaf_3)
        combinations.append(leaf_extract)
        
        concatenation = torch.cat(combinations, 1)
        
        return concatenation

class Inception_Comp3(nn.Module):
    def __init__(self, input_channels, additional_kernel_channels):
        super(Inception_Comp3, self).__init__()
        
        self.leaf1_3_3 = conv_2D_overwrite(input_channels, additional_kernel_channels, kernel_size = 1)
        self.leaf2_3_3 = conv_2D_overwrite(additional_kernel_channels, additional_kernel_channels, kernel_size = (7,1), padding = (3,0))
        self.leaf3_3_3 = conv_2D_overwrite(additional_kernel_channels, additional_kernel_channels, kernel_size = (1,7), padding = (0,3))
        self.leaf4_3_3 = conv_2D_overwrite(additional_kernel_channels, additional_kernel_channels, kernel_size = (7,1), padding = (3,0))
        self.leaf5_3_3 = conv_2D_overwrite(additional_kernel_channels, 128, kernel_size = (1,7), padding = (0,3))
        
        self.leaf_main = conv_2D_overwrite(input_channels, 128, kernel_size = 1)
        
        self.leaf_extract = conv_2D_overwrite(input_channels, 128, kernel_size = 1)
        
        self.leaf1 = conv_2D_overwrite(input_channels, additional_kernel_channels, kernel_size = 1)
        self.leaf2 = conv_2D_overwrite(additional_kernel_channels, additional_kernel_channels, kernel_size = (1,7), padding = (0,3))
        self.leaf3 = conv_2D_overwrite(additional_kernel_channels, 128, kernel_size = (7,1), padding = (3,0))
        
    def forward(self, x):
        
        leaf_extract = F.avg_pool2d(x, kernel_size = 3, stride = 1, padding = 1)
        leaf_extract = self.leaf_extract(leaf_extract)
        
        leaf_3 = self.leaf1(x)
        leaf_3 = self.leaf2(leaf_3)
        leaf_3 = self.leaf3(leaf_3)
        
        leaf_1 = self.leaf_main(x)
        
        leaf_5 = self.leaf1_3_3(x)
        leaf_5 = self.leaf2_3_3(leaf_5)
        leaf_5 = self.leaf3_3_3(leaf_5)
        leaf_5 = self.leaf4_3_3(leaf_5)
        leaf_5 = self.leaf5_3_3(leaf_5)
        
        combinations = []
        
        combinations.append(leaf_1)
        combinations.append(leaf_3)
        combinations.append(leaf_5)
        combinations.append(leaf_extract)
        
        concatenation = torch.cat(combinations, 1)
        
        return concatenation

class Inception_Comp4(nn.Module):
    def __init__(self, input_channels):
        super(Inception_Comp4, self).__init__()
        
        self.leaf1_3_3 = conv_2D_overwrite(input_channels, 32, kernel_size = 1)
        self.leaf2_3_3 = conv_2D_overwrite(32, 64, kernel_size = (1,7), padding = (0,3))
        self.leaf3_3_3 = conv_2D_overwrite(64, 128, kernel_size = (7,1), padding = (3,0))
        self.leaf4_3_3 = conv_2D_overwrite(128, 192, kernel_size = 3, stride = 2)
        
        self.leaf1 = conv_2D_overwrite(input_channels, 32, kernel_size = 1)
        self.leaf2 = conv_2D_overwrite(32, 64, kernel_size = 3, stride = 2)
        
    def forward(self, x):
        
        leaf_extract = F.max_pool2d(x, kernel_size = 3, stride = 2)
        
        leaf_1 = self.leaf1(x)
        leaf_1 = self.leaf2(leaf_1)
        
        leaf_main = self.leaf1_3_3(x)
        leaf_main = self.leaf2_3_3(leaf_main)
        leaf_main = self.leaf3_3_3(leaf_main)
        leaf_main = self.leaf4_3_3(leaf_main)
        
        combinations = []
        
        combinations.append(leaf_1)
        combinations.append(leaf_main)
        combinations.append(leaf_extract)
        
        concatenation = torch.cat(combinations, 1)
        
        return concatenation

class conv_2D_overwrite(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super(conv_2D_overwrite, self).__init__()
        self.convolution = nn.Conv2d(input_channels, output_channels, bias = False, **kwargs)
        
        self.batch_normalization = nn.BatchNorm2d(output_channels, eps = 0.001)
        
    def forward(self, x):
        #print('Here')
        x = self.convolution(x)
        #print('Here1')
        x = self.batch_normalization(x)
        #print('Here2')
        x = F.relu(x, inplace = True)
        return x    

class Inception_Helper(nn.Module):
    def __init__(self, input_channels, number_of_classes):
        super(Inception_Helper, self).__init__()
        self.layer_1 = conv_2D_overwrite(input_channels, 128, kernel_size = 1)
        self.layer_2 = conv_2D_overwrite(128, 512, kernel_size = 5)
        self.layer_2.stddev = 0.01
        self.linear = nn.Linear(512, number_of_classes)
        self.linear.stddev = 0.001
        
    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size = 3, stride = 3)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = torch.flatten(x, 1)
        output = self.linear(x)
        return output

class Inception_v3(nn.Module):
    def __init__(self, number_of_classes):
        super(Inception_v3, self).__init__()
        #self.training = training 
        #self.alteration = alteration
        #self.bool_logits = bool_logits
        self.convolutional_layer_2D = conv_2D_overwrite(input_channels = 3, output_channels = 32, kernel_size = 3, padding = 1)
        self.composite_inception_1 = Inception_Comp1(input_channels = 32, extract_features = 8)
        self.composite_inception_2 = Inception_Comp1(input_channels = 64, extract_features = 72)
        self.composite_inception_3 = Inception_Comp2(128)
        self.composite_inception_4 = Inception_Comp3(input_channels=256, additional_kernel_channels = 64)
        
        #if(bool_logits == True):
        self.classifier_helper = Inception_Helper(512, number_of_classes)
        
        self.composite_inception_5 = Inception_Comp4(512)
        self.classifier_network = nn.Linear(768, number_of_classes)
        #self.classifier_auxillary = Auxillary_Classifier(512, number_of_classes)
        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                stddev = float(m.stddev) if hasattr(m, "stddev") else 0.1  # type: ignore
                torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=stddev, a=-2, b=2)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
    def forward(self, x):
        global helper
        print(x.shape)
        x = x.view(1, 3, 75, 75)
        print(x.shape)
        #print(self.convolutional_layer_2D.weight.shape)
        x = self.convolutional_layer_2D(x)
        #print(x.shape)
        x = self.composite_inception_1(x)
        #print(x.shape)
        x = self.composite_inception_2(x)
        x = self.composite_inception_3(x)
        x = self.composite_inception_4(x)
        #if(self.bool_logits == True and self.training == True):
        helper = self.classifier_helper(x)
        x = self.composite_inception_5(x)
        x_pooled = self.average_pool(x)
        x = F.dropout(x_pooled)
        #flattened_input = x.view(x.shape[0], -1)
        flattened_input = torch.flatten(x, 1)
        self.output = self.classifier_network(flattened_input)
        #if(self.bool_logits == True and self.training):
        return self.output, helper
        #return self.output    


gc.collect()
model_Inception_v3 = Inception_v3(29)
model_Inception_v3.load_state_dict(torch.load('C:/Users/DELL/Desktop/EE541_Project/modelInceptionv3_parameters2.pth'))

def create_window_frame(lower_boundary, input_recording_frame):
    upper_boundary_coord_xy = lower_boundary + 200
    print(type(input_recording_frame))
    window_frame = input_recording_frame[lower_boundary:upper_boundary_coord_xy, lower_boundary:upper_boundary_coord_xy]
    sketched_area = cv2.resize(window_frame, (200,200))
    return sketched_area

web_camera = cv2.VideoCapture(0)
temp_latitude = web_camera.get(4)
temp_longitude = web_camera.get(3)

longitude = int(temp_longitude)
latitude = int(temp_latitude)

video_writer = cv2.VideoWriter('C:/Users/DELL/Desktop/WebCamFrames/captured_frames_real_time_EE541.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (longitude, latitude))
#exit = False

while(web_camera.isOpened() != False):
    #if(exit):
        #break
    _, temp_frame = web_camera.read()
    cv2.rectangle(temp_frame, (100,100), (300,300), (0,255,255), 1)
    captured_frame = create_window_frame(100, temp_frame)
    transformed_image= feed_transformation(captured_frame)  
    transformed_image = transformed_image.to(device)
    #print(transformed_image.shape)
    
    temp_output, _ = model_Inception_v3(transformed_image)
    _, prediction = torch.max(temp_output.data, 1)
    #print(type(prediction.numpy()))
    #print(type(prediction.numpy()))
    #print((prediction.numpy()).shape)
    temp_prediction = prediction.numpy()
    #print(temp_prediction[0])
    temp_letter = convert_label_to_class(temp_prediction[0])
    #print(temp_letter)
    cv2.putText(temp_frame, temp_letter, (25, 75), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    #str_pred = '' + str(int(prediction))
    #cv2.putText(temp_frame, str_pred ,(25, 75), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    cv2.imshow('transformed_image', temp_frame)
    video_writer.write(temp_frame)

    if(0xFF == ord('e') & cv2.waitKey(27)):
        break
        #exit = True

web_camera.release
cv2.destroyAllWindows()