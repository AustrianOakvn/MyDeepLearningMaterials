# Dataloader
# Network
# Loss
# Optimizer
# Training, validataion

from pyparsing import col
from lib import *
from make_data_path import make_datapath_list
from dataset import MyDataset, my_collate_fn
from transform import DataTransform
from extract_inform_annotation import Anno_xml
from model import SSD
from multiboxloss import MultiBoxLoss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
# print(device)

# dataloader
root_path = root_path = "/mnt/30A83E93A83E5794/Projects/VOCdevkit/VOC2008"
train_img_list, train_anno_list, valid_img_list, valid_anno_list = make_datapath_list(root_path=root_path)
classes = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"] 
color_mean = [104, 117, 123]
input_size = 300

train_dataset = MyDataset(train_img_list, train_anno_list, 
                            phase="train", transform=DataTransform(input_size, color_mean), 
                            anno_xml=Anno_xml(classes))

val_dataset = MyDataset(train_img_list, train_anno_list, 
                            phase="val", transform=DataTransform(input_size, color_mean), 
                            anno_xml=Anno_xml(classes))


batch_size = 8

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)

val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)

dataloader_dict = {"train": train_dataloader,
                    "val": val_dataloader}

# Network
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

model = SSD(phase="train", cfg=cfg)
vgg_weights = torch.load("SSD/vgg_pretrained/vgg16_reducedfc.pth")
model.vgg.load_state_dict(vgg_weights)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


model.extras.apply(weights_init)
model.loc.apply(weights_init)
model.conf.apply(weights_init)

# print(model)

# Multibox Loss
criterion = MultiBoxLoss(jaccard_threshold=0.5, neg_pos=3, device=device)

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

# training and validation

def train_model(net, dataloader_dict, criterion, optimizer, num_epochs):
    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []
    net.to(device)

    for epoch in range(num_epochs+1):
        t_epoch_start = time.time()
        t_iter_start = time.time()

        print("---"*20) 
        print("Epoch{}/{}".format(epoch+1, num_epochs))
        print("---"*20)

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
                print("Training")
            else:
                # Validate every 10 epoch
                if epoch+1 %10 == 0:
                    net.eval()
                    print("---"*10)
                    print("Validation")
                else:
                    continue
            for images, targets in dataloader_dict[phase]:
                images = images.to(device)
                targets = [ann.to(device) for ann in targets]

                # init optimizer
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase=="train"):
                    outputs = net(images)

                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    if phase == "train":
                        loss.backward()

                        nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)

                        optimizer.step() #update params

                        if iteration % 10 == 0:
                            t_iter_end = time.time()
                            duration = t_iter_end - t_iter_start
                            print("Iteration: {} || Loss: {:.4f} || 10iter: {:.4f} sec".format(iteration, loss.item(), duration))

                            t_iter_start = time.time()

                        epoch_train_loss += loss.item()
                        iteration += 1
                    else:
                        epoch_val_loss += loss.item()
        t_epoch_end = time.time()
        print("---"*20)
        print("Epoch {} || epoch train loss: {:.4f} || Epoch val loss: {:.4f}".format(epoch, epoch_train_loss, epoch_val_loss))
        print("Duration: {:.4f} sec".format(t_epoch_end - t_epoch_start))
        t_epoch_start = time.time()

        log_epoch = {"epoch": epoch+1, "train_loss":epoch_train_loss, "val_loss":epoch_val_loss}

        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("SSD/logs/ssd_log.csv")

        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        if (epoch+1) % 10 == 0:
           torch.save(net.state_dict(), "SSD/weights/ssd_300_epoch"+str(epoch+1)+".pth") 





if __name__ == "__main__":
    # test = np.random.rand(3, 4)
    # np.savez_compressed("SSD/weights/test_path.npz", test)
    num_epoch = 30
    train_model(model, dataloader_dict, criterion, optimizer, num_epoch)