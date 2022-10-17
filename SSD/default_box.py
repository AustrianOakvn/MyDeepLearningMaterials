from lib import *


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

class DefaultBox():
    def __init__(self, cfg) -> None:
        self.img_size = cfg["input_size"]
        self.feature_maps = cfg["feature_maps"]
        self.min_size = cfg["min_size"]
        self.max_size = cfg["max_size"]
        self.aspect_ratios = cfg["aspect_ratios"]
        self.steps = cfg["steps"]


    def create_defbox(self):
        defbox_list = []
        for k, f in enumerate(self.feature_maps):
            for i, j in itertools.product(range(f), repeat=2):
                # Number of smaller grid in the feature map
                f_k = self.img_size / self.steps[k]
                # Center of a grid cell
                cx = (i+0.5)/f_k
                cy = (j+0.5)/f_k
                # Small box (square)
                s_k = self.min_size[k]/self.img_size
                defbox_list += [cx, cy, s_k, s_k]

                # Big box (square)
                s_k_ = sqrt(s_k*self.max_size[k]/self.img_size)
                defbox_list += [cx, cy, s_k_, s_k_]

                # Rectangle
                for ar in self.aspect_ratios[k]:
                    defbox_list += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    defbox_list += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        
        # Reshape tensor to (8732, 4)
        output = torch.Tensor(defbox_list).view(-1, 4)
        # Limit the default box to be in range 0, 1 because some default box at top position may lay outside
        output.clamp_(max=1, min=0)
        return output

if __name__ == "__main__":
    defbox = DefaultBox(cfg)
    dbox_list = defbox.create_defbox()
    print(pd.DataFrame(dbox_list.numpy()))