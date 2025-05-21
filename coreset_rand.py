import os
import torch
import numpy as np
import argparse
import pickle

import torch.distributed as dist

from config.config import config as cfg

def main():
    dist.init_process_group(backend='nccl', init_method='env://')
    np.random.seed(73)

    save_path = cfg.coreset_output + '_rand/' + cfg.coreset_order
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # assign each sample to its label if necessary
    with open('/data/mcaldeir/output/labels/label_dict.pkl', 'rb') as fp:
        label_dict = pickle.load(fp)

    keep = []
    for id in range(0, cfg.num_classes):
        samples=label_dict[id]

        kept_amount = int(len(samples) * cfg.fraction)
        if kept_amount < cfg.min_samples_per_id:
            if len(samples) > cfg.min_samples_per_id:
                kept_amount = cfg.min_samples_per_id
            else:
                kept_amount = len(samples)

        keep += list(set(samples[-kept_amount:]))
    keep=sorted(keep)

    # save final labels to a .py file
    file_content = f"kept_index = {keep}"
    with open(save_path+'/'+cfg.dataset+'_'+str(int(cfg.fraction*100))+'.txt', "w") as file:
        file.write(file_content)

    print(f'Finished selecting {cfg.fraction * 100}% ({len(keep)} samples) of the ids of the {cfg.dataset} dataset ({cfg.num_image} samples)')

if __name__ == "__main__":
    main()