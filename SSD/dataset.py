from lib import *
from make_data_path import make_datapath_list
from transform import DataTransform
from extract_inform_annotation import Anno_xml

class MyDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform, anno_xml) -> None:
        super().__init__()
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        # Class transform
        self.transform = transform
        # Class anno_xml
        self.anno_xml = anno_xml


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, index):
        img, gt, height, width = self.pull_item(index)
        return img, gt

    def pull_item(self, index):
        img_file_path = self.img_list[index]
        img = cv2.imread(img_file_path) # BGR
        height, width, channels = img.shape

        # Get annotation
        anno_file_path = self.anno_list[index]
        anno_info = self.anno_xml(anno_file_path, width, height)

        # preprocess
        img, boxes, labels = self.transform(img, self.phase, anno_info[:,:4], anno_info[:, 4])

        # BGR -> RGB, (height, width, channels) --> (channels, height, width)
        img = torch.from_numpy(img[:,:, (2, 1, 0)]).permute(2, 0, 1)

        # Ground truth
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width



def my_collate_fn(batch):
    '''Because each image has different amount of targets
    If not use collate function then tensor of targets for each images would have
    different shape and they cannot be stacked'''

    targets = []
    imgs = []

    for sample in batch:
        imgs.append(torch.FloatTensor(sample[0])) # sample[0] is the image
        targets.append(torch.FloatTensor(sample[1]))

    # Convert to torch tensor
    # list -> (batch_size, 3, 300, 300)
    imgs = torch.stack(imgs, dim=0)

    return imgs, targets




if __name__ == "__main__":
    classes = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"] 
    root_path = "/mnt/30A83E93A83E5794/Projects/VOCdevkit/VOC2008"
    train_img_list, train_annotation_list, valid_img_list, valid_annotation_list = make_datapath_list(root_path=root_path)

    color_mean = [104, 117, 123]
    transformation = DataTransform(300, color_mean)
    trans_anno = Anno_xml(classes)

    train_dataset = MyDataset(train_img_list, train_annotation_list, phase="train", transform=transformation, anno_xml=trans_anno)
    valid_dataset = MyDataset(valid_img_list, valid_annotation_list, phase="val", transform=transformation, anno_xml=trans_anno)

    # print(len(train_dataset))
    # img, gt = train_dataset.__getitem__(1)
    # print(train_dataset.__getitem__(1))
    # print(gt)
    # print(img.shape)


    batch_size = 4
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)

    # train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # valid_dataloader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    dataloader_dict = {
        "train": train_dataloader,
        "valid": valid_dataloader
    }

    batch_iter = iter(dataloader_dict["valid"])
    images, targets = next(batch_iter) # get 1 sample
    print(images.size())
    print(len(targets))
    print(targets[0].size())

