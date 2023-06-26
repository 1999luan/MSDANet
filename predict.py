import os
import time

import PIL
import cv2
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import UNet, UNet_3Plus, AttU_Net, MSDANet, ResUNetPlusPlus, ConvUNeXt, DCSAUNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 1  # exclude background
    weights_path = "./save_weights/model_98_MSDANet.pth"
    img_path = "./1100.png"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    # mean = (0.171, 0.155, 0.190)
    # std = (0.263, 0.241, 0.296)
    #mean = (0.402, 0.402, 0.402)
    #std = (0.298, 0.298, 0.298)
    # mean = (0.331, 0.331, 0.331)
    # std = (0.221, 0.221, 0.221)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    #model = UNet(in_channels=3, num_classes=classes+1, base_c=32)
    #model = ResUNetPlusPlus(in_channels=3, num_classes=2)
    #model = UNet_3Plus(in_channels=3, n_classes=classes + 1)
    #model = ConvUNeXt(in_channels=3, num_classes=2, base_c=32)
    #model = DCSAUNet(img_channels=3, n_classes=2)
    #model = AttU_Net(in_channels=3, num_classes=2)
    model = MSDANet(in_channels=3, num_classes=2, base_c=32)
    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # load image
    original_img = Image.open(img_path).convert('RGB')
    original_img = original_img.resize((480, 480))
    #original_img = cv2.imread(img_path)
    #original_img = cv2.resize(original_img, (240, 240), interpolation=cv2.INTER_AREA)
    #original_img = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 将前景对应的像素值改成255(白色)
        prediction[prediction == 1] = 255
        # 将不敢兴趣的区域像素设置成0(黑色)
        prediction[prediction == 0] = 0
        mask = Image.fromarray(prediction)
        mask.save("./output_image/test.png")


if __name__ == '__main__':
    main()
