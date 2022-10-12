from lib import *
from make_data_path import *


class Anno_xml:
    def __init__(self, classes) -> None:
        '''classes: list of VOC classes'''
        self.classes = classes

    def __call__(self, xml_path, width, height, *args, **kwds):
        ret = []

        # Read xml file
        xml = ET.parse(xml_path).getroot()

        # Iterate over xml to get object inside
        for obj in xml.iter('object'):
            difficult = int(obj.find("difficult").text)
            if difficult == 1:
                continue
            # Bounding box 
            bndb = []
            name = obj.find("name").text.lower().strip()

            # Only find boundingbox at the object level not the part level
            bbox = obj.find("bndbox")

            pts = ["xmin", "ymin", "xmax", "ymax"]
            for pt in pts:
                # Because in VOC coordinate start at (1, 1)
                pixel = int(float(bbox.find(pt).text)) -1

                # Calculate the ratio
                if pt == "xmin" or pt == "xmax":
                    pixel /= width # ratio of width
                else:
                    pixel /= height #ratio of height

                bndb.append(pixel)
            
            label_id = self.classes.index(name)
            bndb.append(label_id)

            ret += [bndb]

        return np.array(ret) # [[xmin, ymin, xmax, ymax, label_id],...]



if __name__ == "__main__":
    classes = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"] 

    anno_xml = Anno_xml(classes=classes)

    root_path = "/mnt/30A83E93A83E5794/Projects/VOCdevkit/VOC2008"
    train_img_list, train_annotation_list, valid_img_list, valid_annotation_list = make_datapath_list(root_path=root_path)

    idx = 0
    img_file_path = train_img_list[idx]
    # custom_anno_pth = r"/mnt/30A83E93A83E5794/Projects/VOCdevkit/VOC2008/Annotations/2007_000027.xml"
    # custom_img_pth = r"/mnt/30A83E93A83E5794/Projects/VOCdevkit/VOC2008/JPEGImages/2007_000027.jpg"


    # Opencv format [height, width, BGR(3 channels)]
    img = cv2.imread(img_file_path)
    height, width, channels = img.shape

    # print(height, width, channels)
    print(train_annotation_list[idx])

    annotation_info = anno_xml(train_annotation_list[idx], width=width, height=height)
    # annotation_info = anno_xml(custom_anno_pth, width=width, height=height)

    print(annotation_info)
