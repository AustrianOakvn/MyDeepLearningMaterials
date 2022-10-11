import torch
import hydra
from omegaconf import OmegaConf


@hydra.main(config_path='Conf', config_name='train')
def main(config):

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = "cpu"

    