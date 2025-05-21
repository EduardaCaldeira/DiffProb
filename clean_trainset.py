import argparse
import logging
import os
import time
import pickle

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss

from utils import losses
from config.config import config as cfg
from utils.dataset import MXFaceDataset, DataLoaderX
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging

from backbones.iresnet import iresnet100, iresnet50


torch.backends.cudnn.benchmark = True

def main(args):
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    cfg.output = cfg.output + '/original/' + cfg.original_net_loss

    if not os.path.exists(cfg.output) and rank == 0:
        os.makedirs(cfg.output)
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output)

    trainset = MXFaceDataset(root_dir=cfg.rec, label_map_root_path=None, order=None, method=None, fraction=1, threshold=None, file_path=None, local_rank=local_rank)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True)
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=trainset, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=0, pin_memory=True, drop_last=False)

    # load model
    if cfg.network == "iresnet100":
        backbone = iresnet100(num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    elif cfg.network == "iresnet50":
        backbone = iresnet50(dropout=0.4,num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    else:
        backbone = None
        logging.info("load backbone failed!")
        exit()

    eval_step = int(len(trainset) / cfg.batch_size / 2)

    if args.eval_epoch is not None:
        cfg.eval_epoch = args.eval_epoch

    resume_step = eval_step * cfg.eval_epoch

    try:
        backbone_pth = os.path.join(cfg.output, str(resume_step) + "backbone.pth")
        backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))

        if rank == 0:
            logging.info("backbone resume loaded successfully!")
    except (FileNotFoundError, KeyError, IndexError, RuntimeError):
        logging.info("load backbone resume init, failed!")

    for ps in backbone.parameters():
        dist.broadcast(ps, 0)

    backbone = DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank])
    
    header = losses.CosFace(in_features=cfg.embedding_size, out_features=cfg.num_classes_original, s=cfg.s, m=cfg.m).to(local_rank)
    header_no_margin = losses.CosFace(in_features=cfg.embedding_size, out_features=cfg.num_classes_original, s=cfg.s, m=0).to(local_rank)
    
    try:
        header_pth = os.path.join(cfg.output, str(resume_step) + "header.pth")
        header.load_state_dict(torch.load(header_pth, map_location=torch.device(local_rank)))
        header_no_margin.load_state_dict(torch.load(header_pth, map_location=torch.device(local_rank)))

        if rank == 0:
            logging.info("header resume loaded successfully!")
    except (FileNotFoundError, KeyError, IndexError, RuntimeError):
        logging.info("header resume init, failed!")
    
    header = DistributedDataParallel(
        module=header, broadcast_buffers=False, device_ids=[local_rank])
    header_no_margin = DistributedDataParallel(
        module=header_no_margin, broadcast_buffers=False, device_ids=[local_rank])

    # create directory where the predicted probabilities history will be saved
    os.makedirs(cfg.output + '/pred_prob_clean', exist_ok=True)

    pred_prob = torch.zeros(cfg.num_image).cuda()
    backbone.eval()
    header.eval()
    header_no_margin.eval()
    train_sampler.set_epoch(cfg.eval_epoch)
    with torch.no_grad():
        for _, (img, label, index) in enumerate(train_loader): 
            img = img.cuda(local_rank, non_blocking=True)
            label = label.cuda(local_rank, non_blocking=True)

            features = F.normalize(backbone(img))
            thetas = header(features, label)
            thetas_no_margin = header_no_margin(features, label)

            pred_label = torch.argmax(F.softmax(thetas_no_margin, dim=1), dim=1)    

            prob = F.softmax(thetas, dim=1).gather(dim=1, index=label.unsqueeze(1)).squeeze(-1).float() # probability assigned to the GT class for each sample in the batch (size=[batch_size])
            mask = pred_label != label


            prob[mask] = -1
            pred_prob = pred_prob.scatter(dim=0, index=index, src=prob) # probability determined for the GT class is assigned to the respective sample index

        dist.all_reduce(pred_prob, op=dist.ReduceOp.MAX)
        np.savetxt(cfg.output + f'/pred_prob_clean/prob_{cfg.eval_epoch}.txt', pred_prob.cpu().detach().numpy())

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch margin penalty loss  training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--resume', type=int, default=0, help="resume training")
    parser.add_argument('--eval_epoch', type=int, default=None, help="epoch to be analyzed")
    args_ = parser.parse_args()
    main(args_)