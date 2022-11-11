# Jaccard: intersection over union
# Hard negative mining
# Regression loss: MSE -> SmoothL1
from statistics import variance
from lib import *

class MultiBoxLoss(nn.Module):
    def __init__(self, jaccard_threshold=0.5, neg_pos=3, device='cpu') -> None:
        super(MultiBoxLoss, self).__init__()
        self.jaccard_threshold = jaccard_threshold
        self.neg_pos = neg_pos
        self.device = device

    def forward(self, predictions, targets):
        loc_data, conf_data, dbox_list = predictions
        # loc_data format (batch, num_bbox, coord)
        num_batch = loc_data.size(0)
        num_dbox = loc_data.size(1)
        ## Something wrong here
        num_classes = conf_data.size(2)

        # Create 2 empty tensor
        conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self.device)
        loc_t = torch.Tensor(num_batch, num_dbox, 4).to(self.device)

        for idx in range(num_batch):
            truths = targets[idx][:, :-1].to(self.device) # (xmin, ymin, xmax, ymax)
            labels = targets[idx][:, -1].to(self.device)

            dbox = dbox_list.to(self.device)
            variances = [0.1, 0.2]
            match(self.jaccard_threshold, truths=truths, 
                    priors=dbox, variances=variances, 
                    labels=labels ,loc_t=loc_t, 
                    conf_t=conf_t_label, idx=idx)

        pos_mask = conf_t_label > 0
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)

        # positive dbox, loc_data
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

        loss_loc = F.smooth_l1_loss(loc_p, loc_t, reduction="sum")

        # loss conf
        batch_conf = conf_data.view(-1, num_classes) # (batch, num_box, 21) -> (batch*num_box, 21)

        loss_conf = F.cross_entropy(batch_conf, conf_t_label.view(-1), reduction="none")

        # Hard negative mining
        num_pos = pos_mask.sum(1, keepdim=True)
        loss_conf = loss_conf.view(num_batch, -1)

        _, loss_idx = loss_conf.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        #idx rank chính là thông số để biết được độ lớn loss nằm ở vị trí bao nhiêu
        # 
        num_neg = torch.clamp(num_pos*self.neg_pos, max=num_dbox)

        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        # (num_batch, 8732) -> (num_batch, 8732, 21)
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        conf_t_pre = conf_data[(pos_idx_mask+neg_idx_mask).gt(0)].view(-1, num_classes)

        conf_t_label_ = conf_t_label[(pos_mask+neg_mask).gt(0)]

        # loss conf
        loss_conf = F.cross_entropy(conf_t_pre, conf_t_label_, reduction="sum")

        # total loss
        N = num_pos.sum()

        loss_loc = loss_loc/N
        loss_conf = loss_conf/N 

        return loss_loc, loss_conf


if __name__ == "__mainn__":
    