#

from turtle import width
from extract_inform_annotation import Anno_xml
from make_data_path import make_datapath_list
from utils.augmentation import Compose, ConvertFromInts, ToAbsoluteCoords, \
                                PhotometricDistort, Expand, RandomSampleCrop, \
                                RandomMirror, ToPercentCoords, Resize, \
                                SubtractMeans
from lib import *

class DataTransform:
    '''Preprocessing images class'''
    def __init__(self, input_size, color_mean) -> None:
        # Train and val would require different transformation
        self.data_transform = {
            # ConvertFromInts: convert images type to numpy float 32
            "train": Compose([
                            ConvertFromInts(), 
                            # Convert annotation to absolute coord so that we could apply the following transformation to image
                            ToAbsoluteCoords(), # Convert from ratio back to normal size
                            PhotometricDistort(), # Change color by random
                            Expand(color_mean),
                            RandomSampleCrop(),
                            RandomMirror(), # random flip image
                            ToPercentCoords(), # Transform annotation back to [0, 1]
                            Resize(input_size), # default is 300
                            SubtractMeans(color_mean)
                            ]), 
            "val": Compose([
                            ConvertFromInts(),
                            Resize(input_size),
                            SubtractMeans(color_mean)
            ])
        }

    
    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)


if __name__ == "__main__":
    classes = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"] 
    root_path = "/mnt/30A83E93A83E5794/Projects/VOCdevkit/VOC2008"
    train_img_list, train_annotation_list, valid_img_list, valid_annotation_list = make_datapath_list(root_path=root_path)

    img_file_path = train_img_list[0]
    img = cv2.imread(img_file_path)

    height, width, channels = img.shape

    # annotation information
    trans_anno = Anno_xml(classes)
    anno_info_list = trans_anno(train_annotation_list[0], width=width, height=height)

    # Default format of matplotlib is RGB
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    color_mean = [104, 117, 123]
    transformation = DataTransform(300, color_mean)

    phase = "train"
    img_transformed, boxes, labels = transformation(img, phase, anno_info_list[:, :4], anno_info_list[:, 4])
    # plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    # plt.show()
    print(img_transformed.max())
    print(img_transformed)
    print(boxes)
    print(labels)

    phase = "val"
    img_transformed, boxes, labels = transformation(img, phase, anno_info_list[:, :4], anno_info_list[:, 4])
    # plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    # plt.show()


