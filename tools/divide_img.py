
# 将数据集按照，训练集：测试集：验证集 = 8：1：1 划分
import os
import random
import shutil
from shutil import copy2

datadir_img = "F:/xiazai/CVC-ClinicDB/Original"
datadir_mask = "F:/xiazai/CVC-ClinicDB/Ground Truth"
all_img_data = os.listdir(datadir_img)  # （图片文件夹）
all_img_data.sort(key=lambda x:int(x[:-4])) #按照文件名大小顺序进行排序
all_mask_data = os.listdir(datadir_mask)
all_mask_data.sort(key=lambda x:int(x[:-4]))
num_all_data = len(all_img_data)
print("num_all_data: " + str(num_all_data))
index_list = list(range(num_all_data))
# print(index_list)
random.shuffle(index_list)
print(index_list)
num = 0
count = 0

trainImgDir = "F:/xiazai/CVC-ClinicDB/training/images"  # （将训练集放在这个文件夹下）
if not os.path.exists(trainImgDir):
    os.mkdir(trainImgDir)

validImgDir = 'F:/xiazai/CVC-ClinicDB/val/images'  # （将验证集放在这个文件夹下）
if not os.path.exists(validImgDir):
    os.mkdir(validImgDir)

testImgDir = 'F:/xiazai/CVC-ClinicDB/test/images'  # （将测试集放在这个文件夹下）
if not os.path.exists(testImgDir):
    os.mkdir(testImgDir)

trainMaskDir = "F:/xiazai/CVC-ClinicDB/training/masks"  # （将训练集放在这个文件夹下）
if not os.path.exists(trainMaskDir):
    os.mkdir(trainMaskDir)

validMaskDir = 'F:/xiazai/CVC-ClinicDB/val/masks'  # （将验证集放在这个文件夹下）
if not os.path.exists(validMaskDir):
    os.mkdir(validMaskDir)

testMaskDir = 'F:/xiazai/CVC-ClinicDB/test/masks'  # （将测试集放在这个文件夹下）
if not os.path.exists(testMaskDir):
    os.mkdir(testMaskDir)

for i in index_list:
    fileName = os.path.join(datadir_img, all_img_data[i])
    if num < num_all_data * 0.8:
        # print(str(fileName))
        copy2(fileName, trainImgDir)
    elif num > num_all_data * 0.8 and num < num_all_data * 0.9:
        # print(str(fileName))
        copy2(fileName, validImgDir)
    else:
        copy2(fileName, testImgDir)
    num += 1

for i in index_list:
    fileName = os.path.join(datadir_mask, all_mask_data[i])
    if count < num_all_data * 0.8:
        # print(str(fileName))
        copy2(fileName, trainMaskDir)
    elif count > num_all_data * 0.8 and count < num_all_data * 0.9:
        # print(str(fileName))
        copy2(fileName, validMaskDir)
    else:
        copy2(fileName, testMaskDir)
    count += 1