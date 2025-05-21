import os
import torch
import argparse

import torch.distributed as dist

from config.config import config as cfg
from utils.dataset import MXFaceDataset, DataLoaderX

import pickle

def main(args):
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = args.local_rank

    save_path = 'output/labels'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    trainset = MXFaceDataset(root_dir=cfg.rec, method=None, order=None, fraction=1, file_path=cfg.coreset_output + '_' + cfg.coreset_method + '/' + cfg.coreset_order + '/' + cfg.dataset + '_' + str(int(cfg.fraction * 100)) + '.txt'
            , local_rank=local_rank)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=False)
    train_loader = DataLoaderX(local_rank=local_rank, dataset=trainset, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=0, pin_memory=True, drop_last=False)

    label_dict = {}
    for _, (_, label, index) in enumerate(train_loader):
        for k, v in zip(label, index):
            if k.item() not in label_dict:
                label_dict[k.item()]=v.item()
            elif type(label_dict[k.item()]) == list:
                label_dict[k.item()].append(v.item())
            else: 
                label_dict[k.item()]=[label_dict[k.item()], v.item()]
    
    label_dict=dict(sorted(label_dict.items()))

    with open(save_path+'/label_dict.pkl', 'wb') as fp:
        pickle.dump(label_dict, fp)
        print('Label dictionary saved successfully!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch margin penalty loss training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    args_ = parser.parse_args()
    main(args_)