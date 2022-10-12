import torch.nn as nn
import torch

def create_vgg():
    """Tạo ra mô hình VGG"""
    layers = []
    # Số lượng kênh vào
    in_channels = 3
    # List độ sâu channels, M là max pooling
    cfgs = [64, 64, "M", 128, 128, "M",
            256, 256, 256, "MC", 512, 512, 512, "M",
            512, 512, 512]

    for cfg in cfgs:
        if cfg == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # MC là max pooling giá trị làm tròn lên
        elif cfg == "MC":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=cfg, kernel_size=3, padding=1)
            # Dùng inplace tiết kiệm đc memory
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = cfg
        
        # Lớp max pooling thứ 5
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    # Why padding = 6 and dilation = 6
    conv6 = nn.Conv2d(in_channels =512, out_channels=1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]


    return nn.ModuleList(layers)


def create_extras():
    """Module extras"""
    layers = []
    input_channels = 1024
    
    cfgs = [256, 512, 128, 256, 128, 256, 128, 256]

    layers += [nn.Conv2d(in_channels=input_channels, out_channels=cfgs[0], kernel_size=1)]
    layers += [nn.Conv2d(in_channels=cfgs[0], out_channels=cfgs[1], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(in_channels=cfgs[1], out_channels=cfgs[2], kernel_size=1)]
    layers += [nn.Conv2d(in_channels=cfgs[2], out_channels=cfgs[3], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(in_channels=cfgs[3], out_channels=cfgs[4], kernel_size=1)]
    layers += [nn.Conv2d(in_channels=cfgs[4], out_channels=cfgs[5], kernel_size=3)]
    layers += [nn.Conv2d(in_channels=cfgs[5], out_channels=cfgs[6], kernel_size=1)]
    layers += [nn.Conv2d(in_channels=cfgs[6], out_channels=cfgs[7], kernel_size=3)]

    return nn.ModuleList(layers)






def create_loc_conf(num_classes=21, bbox_ratio_num=[4, 6, 6, 6, 4, 4]):
    """Module đầu ra vị trí và confidence"""
    loc_layers = []
    conf_layers = []

    # Source 1
    # Đầu ra qua filter 3x3 và đi vào vector với số chiều là số class nhân với ratio
    loc_layers += [nn.Conv2d(512, bbox_ratio_num[0]*4, kernel_size=3, padding=1)]
    #conf
    conf_layers += [nn.Conv2d(512, bbox_ratio_num[0]*num_classes, kernel_size=3, padding=1)]

    #Source 2
    loc_layers += [nn.Conv2d(1024, bbox_ratio_num[1]*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, bbox_ratio_num[1]*num_classes, kernel_size=3, padding=1)]

    # Source 3
    loc_layers += [nn.Conv2d(512, bbox_ratio_num[2]*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_ratio_num[2]*num_classes, kernel_size=3, padding=1)]

    # Source 4
    loc_layers += [nn.Conv2d(256, bbox_ratio_num[3]*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_ratio_num[3]*num_classes, kernel_size=3, padding=1)]

    # Source 5
    loc_layers += [nn.Conv2d(256, bbox_ratio_num[4]*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_ratio_num[4]*num_classes, kernel_size=3, padding=1)]

    # Source 6
    loc_layers += [nn.Conv2d(256, bbox_ratio_num[5]*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_ratio_num[5]*num_classes, kernel_size=3, padding=1)]
    
    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)


# Kiểm tra

if __name__ == "__main__":
    vgg = create_vgg()
    my_extras = create_extras()
    loc, conf = create_loc_conf()
    test_layer = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6)
    x = torch.rand((1, 512, 19, 19))
    # print(vgg)
    # print(my_extras)
    print(loc)
    print(conf)
    # print(test_layer(x).shape)
    # print(vgg(x).shape)

