import os
import shutil


def segmentation(root: str):
    imgName = []
    labelName = []
    data_root = os.path.join(root, "Dataset_BUSI_with_GT")
    assert os.path.exists(data_root), f"path '{data_root}' does not exists."
    for file in os.listdir(data_root):
        for i in os.listdir(os.path.join(data_root, file)):
            root = os.path.join(data_root, file)
            file_list = os.listdir(os.path.join(data_root, file))
            if i.endswith(".png") and "mask" not in i:
                fileName = i.split(".")[0]
                count = 0
                for j in file_list:
                    if "mask" in j and fileName in j:
                        count = count + 1
                if count == 1:
                    imgName.append(os.path.join(root, i))
                    labelName.append(os.path.join(root, file_list[file_list.index(i)+1]))
    return imgName, labelName

def copy_file(png_list, root):
    for i in png_list:
        if "normal" not in i:
            img_name = str(png_list.index(i)+1)+'.png'
            shutil.copy(i, os.path.join(root, img_name))


if __name__ == '__main__':
    imgName, labelName = segmentation("F:/xiazai")
    copy_file(imgName, "F:/xiazai/111/training/images")
    copy_file(labelName, "F:/xiazai/111/training/masks")