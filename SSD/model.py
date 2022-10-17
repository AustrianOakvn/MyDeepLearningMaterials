# import torch.nn as nn
# import torch

from lib import *
from L2_norm import *
from default_box import *

cfg = {
    "num_classes":21, #20 class + 1 background
    "input_size":300,
    "bbox_aspect_num":[4, 6, 6, 6, 4, 4], # Number of bbox ratio from source 1 -> source 6
    "feature_maps":[38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 100, 300], # size of default box
    "min_size":[30, 60, 111, 162, 213, 264], # What is the meaning of this ?
    "max_size":[60, 111, 162, 213, 264, 315],
    "aspect_ratios":[[2], [2, 3], [2, 3], [2, 3], [2], [2]] # One number equal to 2 boxes
}

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


class SSD(nn.Module):
    def __init__(self, phase, cfg) -> None:
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = cfg["num_classes"]
        # create main modules

        self.vgg = create_vgg()
        self.extras = create_extras()
        self.loc, self.conf = create_loc_conf(cfg["num_classes"], cfg["bbox_aspect_num"])
        self.L2_norm = L2Norm()

        self.dbox = DefaultBox(cfg)
        self.dbox_list = self.dbox.create_defbox()

        if phase == "inference":
            self.detect = Detect()



def decode(loc, def_box_list):
    '''Get bounding box from the default box 
    and the offset from the loc layer of the model
    loc: (8732, 4) each element (delta_cx, delta_cy, delta_w, delta_h)
    def_box_list: (8732, 4) each element (cx_d, cy_d, w_d, h_d)
    
    returns:
    boxes [xmin, ymin, xmax, ymax]'''

    boxes = torch.cat((
                        def_box_list[:, :2] + 0.1*loc[:,:2]*def_box_list[:,2:],
                        def_box_list[:,2:]*torch.exp(loc[:, 2:]*0.2)), dim=1)

    boxes[:, :2] -= boxes[:,2:]/2 # x_min, y_min = cx, cy - w/2, h/2
    boxes[:, 2:] += boxes[:, :2] # x_max, y_max = x_min, y_min + w, h

    return boxes


def nms(boxes, scores, overlap=0.45, top_k=200):
    '''Non maximum suppression
    From 8732 predictions boxes only takes out top k boxes
    - Arrage the candidates for this class in order of decreasing likelihood
    - Consider the candidate with the highest score. Eliminate all candidates with
    lesser scores that have a Jaccard overlap of more than, say, 0.5, with this example
    - Consider the next highest-scoring candiate still remaining in the pool. Eliminate
    all candidates with lesser score that have a Jaccard overlap more than 0.5
    - Repeat until you run through the entire sequence of candidates
    '''
    count = 0
    # Create new tensor with shape equal to scores
    keep = scores.new(scores.size(0)).zero_().long()
    # Boxes coordinate
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = torch.mul(x2-x1, y2-y1)

    tmp_x1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    value, idx = scores.sort(dim=0) #Ascending
    # Take 200 last elements
    idx = idx[-top_k:]

    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1

        if idx.size(0) == 1:
            break
        # New idx without the last element
        idx = idx[:-1]

        # Get all the coordinate with index in idx tensor and place them in tmp value
        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)

        # Take the x_min of the overlap
        # Clamp the value to get the x1, y1, x2, y2 of the overlapped area
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i]) # = x1[i] if tmp_x1 < x1[i]
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x1, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y1, max=y2[i])


        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        # Calculate the width and height of overlaaped boxes
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        # overlapped area
        inter = tmp_w * tmp_h
        others_area = torch.index_select(area, 0, idx)
        union = area[i] + others_area - inter

        # iou of one box vs the rest
        iou = inter/union

        # keep the index with overlap < 0.45
        idx = idx[iou.le(overlap)]

    return keep, count


class Detect(Function):
    '''Function has auto forward function'''

    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Why dim = -1
        self.softmax = nn.Softmax(dim=-1)
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh

    def forward(self, loc_data, conf_data, dbox_list):
        num_batch = loc_data.size(0)
        num_dbox = loc_data.size(1) # 8732 boxes
        num_classes = conf_data.size(2) # 21 classes

        # Convert to probability using softmax
        conf_data = self.softmax(conf_data) 
        # Format (batch_num, num_box, num_class) -> (batch_num, num_class, num_dbox)
        conf_predict = conf_data.transpose(2, 1)

        # Process each image in one batch
        for i in range(num_batch):
            # Calculate bbox from offset and default boxes
            decode_boxes = decode(loc_data[i], dbox_list)
            # Copy confidence score of image i-th
            conf_scores = conf_predict[i].clone()

            for cl in range(1, num_classes):
                c_mask = conf_predict[cl].gt(self.conf_thresh) # only take confidence > 0.01
                scores = conf_predict[cl][c_mask]

                if scores.numel() == 0:
                    continue
                
                # convert back to the dimension of decode_box
                l_mask = c_mask.unsqueeze(1).expand_as(decode_boxes)




# Kiểm tra
if __name__ == "__main__":
    # vgg = create_vgg()
    # my_extras = create_extras()
    # loc, conf = create_loc_conf()
    # test_layer = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6)
    # x = torch.rand((1, 512, 19, 19))
    # print(vgg)
    # print(my_extras)
    # print(loc)
    # print(conf)
    # print(test_layer(x).shape)
    # print(vgg(x).shape)

    ssd = SSD(phase="train", cfg=cfg)
    print(ssd)
