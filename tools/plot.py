from matplotlib import pyplot as plt
import numpy as np
import openpyxl as op

# #传递epoch，和结果路径，绘制结果图
# def plot_img(epoch=100, path='results20220614-101316_unet.txt'):
#     file = open('../'+path)
#     data = file.readlines()
#     train_loss = []
#     train_lr = []
#     dice = []
#     global_correct = []
#     mean_iou = []
#     for i in range(1, len(data), 9):
#         data_strip = data[i].strip("\n")
#         train_loss.append(data_strip.split(":")[1].strip())
#     train_loss = [float(n) for n in train_loss]
#     #print(train_loss)
#
#     for i in range(2, len(data), 9):
#         data_strip = data[i].strip("\n")
#         train_lr.append(data_strip.split(":")[1].strip())
#     train_lr = [float(n) for n in train_lr]
#     #print(train_lr)
#
#     for i in range(3, len(data), 9):
#         data_strip = data[i].strip("\n")
#         dice.append(data_strip.split(":")[1].strip())
#     dice = [float(n) for n in dice]
#     #print(dice)
#
#     for i in range(4, len(data), 9):
#         data_strip = data[i].strip("\n")
#         global_correct.append(data_strip.split(":")[1].strip())
#     global_correct = [float(n) for n in global_correct]
#     #print(global_correct)
#
#     for i in range(7, len(data), 9):
#         data_strip = data[i].strip("\n")
#         mean_iou.append(data_strip.split(":")[1].strip())
#     mean_iou = [float(n) for n in mean_iou]
#     #print(mean_iou)
#
#     x = np.arange(1, epoch+1)
#     plt.xticks(np.arange(0, epoch+1, 10)) #设置x坐标轴间距
#     plt.plot(x, train_loss, color='r', label='loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('loss')
#     plt.title('损失变化')
#     plt.legend()
#     #plt.grid()
#     plt.show()
#
#     plt.xticks(np.arange(0, epoch+1, 10))
#     plt.plot(x, train_lr, color='g', label='lr')
#     plt.xlabel('Epoch')
#     plt.ylabel('lr')
#     plt.title('学习率变化')
#     plt.legend()
#     #plt.grid()
#     plt.show()
#
#     plt.xticks(np.arange(0, epoch+1, 10))
#     plt.plot(x, dice, color='y', label='dice')
#     plt.xlabel('Epoch')
#     plt.ylabel('dice')
#     plt.title('dice分数')
#     plt.legend()
#     # plt.grid()
#     plt.show()
#
#     plt.xticks(np.arange(0, epoch+1, 10))
#     plt.plot(x, global_correct, color='blue', label='global_correct')
#     plt.xlabel('Epoch')
#     plt.ylabel('global_correct')
#     plt.title('平均正确率')
#     plt.legend()
#     # plt.grid()
#     plt.show()
#
#     plt.xticks(np.arange(0, epoch+1, 10))
#     plt.plot(x, mean_iou, color='pink', label='mean_iou')
#     plt.xlabel('Epoch')
#     plt.ylabel('mean_iou')
#     plt.title('平均Iou')
#     plt.legend()
#     # plt.grid()
#     plt.show()

# 数据处理
def data_process(path='cvc_lr_0.01_dice_0.9142.txt'):
    file = open('../'+path)
    data = file.readlines()
    train_loss = []
    train_lr = []
    obj_dice = []
    background_dice = []
    mean_dice = []
    global_correct = []
    target_accuracy = []
    target_iou = []
    mean_iou = []
    data_source = []
    precision = []
    test_loss = []
    for i in range(1, len(data), 13):
        data_strip = data[i].strip("\n")
        train_loss.append(data_strip.split(":")[1].strip())
    train_loss = [float(n) for n in train_loss]
    data_source.append(train_loss)

    for i in range(2, len(data), 13):
        data_strip = data[i].strip("\n")
        train_lr.append(data_strip.split(":")[1].strip())
    train_lr = [float(n) for n in train_lr]
    data_source.append(train_lr)

    for i in range(3, len(data), 13):
        data_strip = data[i].strip("\n")
        obj_dice.append(data_strip.split(":")[1].strip())
    obj_dice = [float(n) for n in obj_dice]
    data_source.append(obj_dice)

    for i in range(4, len(data), 13):
        data_strip = data[i].strip("\n")
        background_dice.append(data_strip.split(":")[1].strip())
    background_dice = [float(n) for n in background_dice]
    data_source.append(background_dice)

    for i in range(5, len(data), 13):
        data_strip = data[i].strip("\n")
        mean_dice.append(data_strip.split(":")[1].strip())
    mean_dice = [float(n) for n in mean_dice]
    data_source.append(mean_dice)

    for i in range(6, len(data), 13):
        data_strip = data[i].strip("\n")
        global_correct.append(data_strip.split(":")[1].strip())
    global_correct = [float(n) for n in global_correct]
    data_source.append(global_correct)

    for i in range(7, len(data), 13):
        data_strip = data[i].strip("\n")
        if data_strip.split(":")[1].strip()[2:5] == '100':
            target_accuracy.append(data_strip.split(":")[1].strip()[12:-2])
        else:
            target_accuracy.append(data_strip.split(":")[1].strip()[11:-2])
    target_accuracy = [float(n) for n in target_accuracy]
    data_source.append(target_accuracy)

    for i in range(8, len(data), 13):
        data_strip = data[i].strip("\n")
        if data_strip.split(":")[1].strip()[2:5] == '100':
            precision.append(data_strip.split(":")[1].strip()[12:-2])
        else:
            precision.append(data_strip.split(":")[1].strip()[11:-2])
    precision = [float(n) for n in precision]
    data_source.append(precision)

    for i in range(9, len(data), 13):
        data_strip = data[i].strip("\n")
        if data_strip.split(":")[1].strip()[2:5] == '100':
            target_iou.append(data_strip.split(":")[1].strip()[12:-2])
        else:
            target_iou.append(data_strip.split(":")[1].strip()[11:-2])
    target_iou = [float(n) for n in target_iou]
    data_source.append(target_iou)

    for i in range(10, len(data), 13):
        data_strip = data[i].strip("\n")
        mean_iou.append(data_strip.split(":")[1].strip())
    mean_iou = [float(n) for n in mean_iou]
    data_source.append(mean_iou)

    for i in range(11, len(data), 13):
        data_strip = data[i].strip("\n")
        test_loss.append(data_strip.split(":")[1].strip())
    test_loss = [float(n) for n in test_loss]
    data_source.append(test_loss)

    return data_source

# 根据数据进行画图
# def plot_data(data1_para, epoch=400, ylable='dice', name=' dice分数'):
#     plt.figure()
#     x = np.arange(1, epoch + 1)
#     plt.xticks(np.arange(0, epoch + 1, 40))
#     plt.plot(x, data1_para, color='r', label='MSDAUnet')
#     plt.xlabel('Epoch')
#     plt.ylabel(ylable)
#     plt.title(name)
#     plt.legend()
#     # plt.grid()
#     plt.show()

def plot_data(data1_para1, data_para2, epoch=400, ylable='Loss'):
    plt.figure()
    x = np.arange(1, epoch + 1)
    plt.xticks(np.arange(0, epoch + 1, 40))
    plt.plot(x, data1_para1, color='r', label='train')
    plt.plot(x, data_para2, color='g', label='val')
    plt.xlabel('Epoch')
    plt.ylabel(ylable)
    plt.legend()
    # plt.grid()
    plt.show()
# def plot_data(data1_para, data2_para, data3_para, data4_para, data5_para, epoch=100, ylable='dice', name=' dice分数'):
#     x = np.arange(1, epoch + 1)
#     plt.xticks(np.arange(0, epoch + 1, 10))
#     plt.plot(x, data1_para, color='r', label='LZ_Unet')
#     plt.plot(x, data2_para, color='b', label='Unet')
#     plt.plot(x, data3_para, color='g', label='Unet_2plus')
#     plt.plot(x, data4_para, color='orange', label='att_Unet')
#     plt.plot(x, data5_para, color='grey', label='Unet_3plus')
#     plt.xlabel('Epoch')
#     plt.ylabel(ylable)
#     plt.title(name)
#     plt.legend()
#     # plt.grid()
#     plt.show()


#绘制实验对比图
def plot_compare(epoch=400, path1='MSDAUNet_学习率_0.01_dice_0.9142.txt'):
    data1 = data_process(path1)
    plot_data(data1[0], data1[10], epoch)
    # plot_data(data1[0], epoch, 'train_loss', '训练损失')
    # plot_data(data1[2], epoch, 'obj_dice', 'obj_dice分数')
    # plot_data(data1[3], epoch, 'background_dice', 'background_dice分数')
    # plot_data(data1[4], epoch, 'mean_dice', 'mean_dice分数')
    # plot_data(data1[5], epoch, 'global_correct', '准确率')
    # plot_data(data1[6], epoch, 'target_accuracy', '召回率')
    # plot_data(data1[7], epoch, 'precision', '精准率')
    # plot_data(data1[8], epoch, 'target_iou', '目标Iou')
    # plot_data(data1[9], epoch, 'mean_iou', '平均Iou')
    # plot_data(data1[10], epoch, 'test_loss', '测试损失')
# def plot_compare(epoch=100, path1='ISP_net.txt', path2='results20220614-101316_unet.txt', path3='results20220610-100727_2plus.txt', path4='results20220615-153710_att.txt', path5='results20220615-092944_3plus.txt'):
#     data1 = data_process(path1)
#     data2 = data_process(path2)
#     data3 = data_process(path3)
#     data4 = data_process(path4)
#     data5 = data_process(path5)
#     plot_data(data1[0], data2[0], data3[0], data4[0], data5[0], epoch, 'loss', '损失变化')
#     plot_data(data1[2], data2[2], data3[2], data4[2], data5[2], epoch, 'dice', 'dice分数')
#     plot_data(data1[3], data2[3], data3[3], data4[3], data5[3], epoch, 'global_correct', '准确率')
#     plot_data(data1[4], data2[4], data3[4], data4[4], data5[4], epoch, 'target_accuracy', '召回率')
#     plot_data(data1[5], data2[5], data3[5], data4[5], data5[5], epoch, 'precision', '精准率')
#     plot_data(data1[6], data2[6], data3[6], data4[6], data5[6], epoch, 'target_iou', '目标Iou')
#     plot_data(data1[7], data2[7], data3[7], data4[7], data5[7], epoch, 'mean_iou', '平均Iou')
def save_data(path):
    all_data = data_process(path)
    wb = op.load_workbook("C:/Users/luanl/Desktop/DSB对比实验.xlsx")
    sh = wb["Att_Unet"]
    count = 2
    for data in all_data:
        for i, d in enumerate(data):
            sh.cell(i+2, count, d)
        count += 1
    wb.save("C:/Users/luanl/Desktop/DSB对比实验.xlsx")



if __name__ == '__main__':
    plot_compare()
    #print(data_process())
    #save_data("DSB_AttUnet_学习率_0.01.txt")