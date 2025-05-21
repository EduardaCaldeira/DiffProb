import argparse
import logging
import os
import time

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
from utils import mobilefacenet

from backbones.iresnet import iresnet100, iresnet50, iresnet34, iresnet18


torch.backends.cudnn.benchmark = True

def main(args):
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if args.threshold is not None:
        cfg.threshold = args.threshold

    if args.fraction is not None:
        cfg.fraction = args.fraction

    if args.coreset_method is not None:
        cfg.coreset_method = args.coreset_method

    if args.coreset_order is not None:
        cfg.coreset_order = args.coreset_order

    if args.is_original_train is not None:
        cfg.is_original_train = args.is_original_train

    if cfg.coreset_method == 'eval_simprobs_clean':
        cfg.num_classes = 10562

    label_map_root_path = '../../../../data/mcaldeir/data_pruning/original/' + cfg.original_net_loss + '/coresets_' + cfg.coreset_method + '/' + cfg.coreset_order
    if cfg.coreset_method == 'eval_simprobs' or cfg.coreset_method == 'eval_simprobs_clean':
        label_map_root_path = label_map_root_path + '/epoch_' + str(cfg.eval_epoch)

    # defining important paths according to the type of training performed
    if cfg.is_original_train:
        cfg.output = os.path.join(cfg.output, "original", cfg.network+"_"+cfg.loss)
        file_path = None
    else:
        pruned_data_path = os.path.join(cfg.output, "original", cfg.original_net_loss)
        cfg.output = os.path.join(cfg.output, "pruned", cfg.original_net_loss, cfg.pruned_net_loss)

        if cfg.coreset_method =='eval_simprobs' or cfg.coreset_method == 'eval_simprobs_clean':
            cfg.output = cfg.output + '/coresets_' + cfg.coreset_method + '/' + cfg.coreset_order + '/epoch_'+str(cfg.eval_epoch)+'/pruned_'+str(int(cfg.threshold*10000))
            file_path = pruned_data_path + '/coresets_' + cfg.coreset_method + '/' + cfg.coreset_order + '/epoch_' + str(cfg.eval_epoch) + '/' + cfg.dataset + '_' + str(int(cfg.threshold * 10000)) + '.txt'
        else:
            cfg.output = cfg.output + '/coresets_' + cfg.coreset_method+'/' + cfg.coreset_order + '/pruned_'+str(int(cfg.fraction*100))
            file_path = pruned_data_path + '/coresets_' + cfg.coreset_method + '/' + cfg.coreset_order + '/' + cfg.dataset + '_' + str(int(cfg.fraction * 100)) + '.txt'
        
    if not os.path.exists(cfg.output) and rank == 0:
        os.makedirs(cfg.output)
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output)

    logging.info("Margin: {}".format(cfg.m))

    trainset = MXFaceDataset(root_dir=cfg.rec, label_map_root_path=label_map_root_path, order=cfg.coreset_order, method=cfg.coreset_method,
                             fraction=cfg.fraction, threshold=cfg.threshold, file_path=file_path, local_rank=local_rank)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True)
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=trainset, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=0, pin_memory=True, drop_last=True)

    # load model
    if cfg.network == "iresnet100":
        backbone = iresnet100(num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    elif cfg.network == "iresnet50":
        backbone = iresnet50(dropout=0.4,num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    elif cfg.network == "iresnet34":    
        backbone = iresnet34(dropout=0.4, num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    elif cfg.network == "iresnet18":    
        backbone = iresnet18(dropout=0.4, num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    elif cfg.network == "mfn":    
        backbone = mobilefacenet.MobileFaceNet(input_size=[112, 112], embedding_size=cfg.embedding_size).to(local_rank)
    else:
        backbone = None
        logging.info("load backbone failed!")
        exit()

    if args.resume:
        try:
            backbone_pth = os.path.join(cfg.output, str(cfg.global_step) + "backbone.pth")
            backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))

            if rank == 0:
                logging.info("backbone resume loaded successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("load backbone resume init, failed!")

    for ps in backbone.parameters():
        dist.broadcast(ps, 0)

    backbone = DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank])
    backbone.train()

    logging.info("number of classes given to the header: %d" % cfg.num_classes)

    # get header
    if cfg.loss == "ElasticArcFace":
        header = losses.ElasticArcFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m,std=cfg.std).to(local_rank)
    elif cfg.loss == "ElasticArcFacePlus":
        header = losses.ElasticArcFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m,
                                       std=cfg.std, plus=True).to(local_rank)
    elif cfg.loss == "ElasticCosFace":
        header = losses.ElasticCosFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m,std=cfg.std).to(local_rank)
    elif cfg.loss == "ElasticCosFacePlus":
        header = losses.ElasticCosFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m,
                                       std=cfg.std, plus=True).to(local_rank)
    elif cfg.loss == "ArcFace":
        header = losses.ArcFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m).to(local_rank)
    elif cfg.loss == "CosFace":
        header = losses.CosFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m).to(local_rank)
    elif cfg.loss == "CurricularFace":
        header = losses.CurricularFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m).to(local_rank)
    elif cfg.loss == "AdaFace":
        header = losses.AdaFace(embedding_size=cfg.embedding_size, classnum=cfg.num_classes).to(local_rank)
    else:
        print("Header not implemented")
    if args.resume:
        try:
            header_pth = os.path.join(cfg.output, str(cfg.global_step) + "header.pth")
            header.load_state_dict(torch.load(header_pth, map_location=torch.device(local_rank)))

            if rank == 0:
                logging.info("header resume loaded successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("header resume init, failed!")
    
    header = DistributedDataParallel(
        module=header, broadcast_buffers=False, device_ids=[local_rank])
    header.train()

    opt_backbone = torch.optim.SGD(
        params=[{'params': backbone.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)
    opt_header = torch.optim.SGD(
        params=[{'params': header.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)

    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=cfg.lr_func)
    scheduler_header = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_header, lr_lambda=cfg.lr_func)        

    criterion = CrossEntropyLoss()

    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size / world_size * cfg.num_epoch)
    eval_step = int(len(trainset) / cfg.batch_size / world_size)

    if rank == 0: logging.info("Total Step is: %d" % total_step)

    if args.resume:
        rem_steps = (total_step - cfg.global_step)
        cur_epoch = cfg.num_epoch - int(cfg.num_epoch / total_step * rem_steps)
        logging.info("resume from estimated epoch {}".format(cur_epoch))
        logging.info("remaining steps {}".format(rem_steps))
        
        start_epoch = cur_epoch
        scheduler_backbone.last_epoch = cur_epoch
        scheduler_header.last_epoch = cur_epoch

        # --------- this could be solved more elegant ----------------
        opt_backbone.param_groups[0]['lr'] = scheduler_backbone.get_lr()[0]
        opt_header.param_groups[0]['lr'] = scheduler_header.get_lr()[0]

        print("last learning rate: {}".format(scheduler_header.get_lr()))
        # ------------------------------------------------------------

    if cfg.is_original_train:
        os.makedirs(cfg.output + '/pred_prob_history', exist_ok=True)

    callback_verification = CallBackVerification(eval_step, rank, cfg.val_targets, cfg.rec)
    callback_logging = CallBackLogging(50, rank, total_step, cfg.batch_size, world_size, writer=None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

    loss = AverageMeter()
    global_step = cfg.global_step

    for epoch in range(start_epoch, cfg.num_epoch):
        train_sampler.set_epoch(epoch)
        pred_prob = torch.zeros(cfg.num_image).cuda()
        for _, (img, label, _) in enumerate(train_loader):
            global_step += 1
            img = img.cuda(local_rank, non_blocking=True)
            label = label.cuda(local_rank, non_blocking=True)

            features = F.normalize(backbone(img))

            if cfg.loss == "AdaFace":
                norm = torch.norm(features, 2, 1, True)
                output = torch.div(features, norm)
                thetas = header(output, norm, label)
            else:
                thetas = header(features, label)

            if cfg.is_original_train:
                # probability assigned to the GT class for each sample in the batch (size=[batch_size])
                prob = F.softmax(thetas, dim=1).gather(dim=1, index=label.unsqueeze(1)).squeeze(-1).float()
                # the probability determined for the GT class during the current epoch is assigned to the respective sample index
                pred_prob = pred_prob.scatter(dim=0, index=index, src=prob)
           
            loss_v = criterion(thetas, label)
            loss_v.backward()

            clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)

            opt_backbone.step()
            opt_header.step()

            opt_backbone.zero_grad()
            opt_header.zero_grad()

            loss.update(loss_v.item(), 1)
            
            callback_logging(global_step, loss, epoch)
            callback_verification(global_step, backbone)

        scheduler_backbone.step()
        scheduler_header.step()

        callback_checkpoint(global_step, backbone, header)

        # gathers results distributed between the GPUs and saves the predicted probabilities for the current epoch
        if cfg.is_original_train:
            dist.all_reduce(pred_prob, op=dist.ReduceOp.MAX)
            np.savetxt(cfg.output + f'/pred_prob_history/prob_{epoch}.txt', pred_prob.cpu().detach().numpy())
            logging.info("Predictions saved for epoch %d" % epoch)

    logging.info("FINAL VERIFICATION!!!")
    callback_verification(global_step, backbone, finished_training=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch margin penalty loss  training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--resume', type=int, default=0, help="resume training")


    parser.add_argument('--threshold', type=float, default=None, help="threshold")
    parser.add_argument('--fraction', type=float, default=None, help="fraction")
    parser.add_argument('--coreset_method', type=str, default=None, help="method")
    parser.add_argument('--coreset_order', type=str, default=None, help="order")
    parser.add_argument('--is_original_train', type=lambda x: x.lower() == 'true', default=None, help="train with or without pruning")

    args_ = parser.parse_args()
    main(args_)