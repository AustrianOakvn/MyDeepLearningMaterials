from lib import *

def make_datapath_list(root_path):
    '''Trả về đường dẫn '''
    image_path_template = osp.join(root_path, "JPEGImages", "%s.jpg")
    annotation_path_template = osp.join(root_path, "Annotations", "%s.xml")

    # FIle train.txt chứa các id của ảnh
    train_id_names = osp.join(root_path, "ImageSets/Main/train.txt")
    valid_id_names = osp.join(root_path, "ImageSets/Main/val.txt")

    train_img_list = []
    train_annotation_list = []

    valid_img_list = []
    valid_annotation_list = []

    # Mở file train.txt duyệt
    for line in open(train_id_names):
        file_id = line.strip() # Xóa kí tự xuống dòng, space
        # Truyền file id vào %s trong template
        img_path = (image_path_template % file_id)
        anno_path = (annotation_path_template % file_id)

        # Put the path into list
        train_img_list.append(img_path)
        train_annotation_list.append(anno_path)

    # Mở file valid.txt duyệt
    for line in open(valid_id_names):
        file_id = line.strip()

        img_path = (image_path_template % file_id)
        anno_path = (annotation_path_template % file_id)

        valid_img_list.append(img_path)
        valid_annotation_list.append(anno_path)

    return train_img_list, train_annotation_list, valid_img_list, valid_annotation_list

    

if __name__ == "__main__":
    root_path = "/mnt/30A83E93A83E5794/Projects/VOCdevkit/VOC2008"
    train_img_list, train_annotation_list, valid_img_list, valid_annotation_list = make_datapath_list(root_path=root_path)

    print(len(train_img_list))
    print(train_img_list[0])