import os
import torch
import numpy as np
import argparse
import re
import pickle

import torch.distributed as dist

from config.config import config as cfg

def main():
    dist.init_process_group(backend='nccl', init_method='env://')
    np.random.seed(73)

    save_path = cfg.coreset_output + '_dynunc/' + cfg.coreset_order
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pred_prob_his = torch.zeros(cfg.num_image, cfg.window).double()

    # sorting the file list by epoch (ascending order)
    pred_prob_history_files = os.listdir(f'{cfg.coreset_input}/pred_prob_history')
    pred_prob_history_files.sort(key=lambda x: int(re.findall("\d+", x)[0]))

    uncertainty_his = []

    for (idx, pred_prob_file) in enumerate(pred_prob_history_files):
        # transfering the information of the next file (ascending order)
        pred_prob = torch.from_numpy(np.loadtxt(f'{cfg.coreset_input}/pred_prob_history/{pred_prob_file}'))
        indices = pred_prob.nonzero().squeeze().cpu()

        # adds the new epoch probabilities at the end of pred_prob_his while removing the oldest entry
        pred_prob_his[indices] = torch.cat((pred_prob_his[indices, 1:], torch.unsqueeze(pred_prob[indices], 1).cpu()), dim=1)

        # determining uncertainty for the window whose last epoch is the current one
        if idx >= cfg.window - 1 and idx < len(pred_prob_history_files) - 1: # discards incomplete windows
            uncertainty_his.append((torch.std(pred_prob_his, dim=1) * 10).detach().numpy()) 

    # the final uncertainty is determined as the average accross all windows
    dynamic_uncertainty = np.mean(np.array(uncertainty_his), axis=0)

    
    with open('/data/mcaldeir/output/labels/label_dict.pkl', 'rb') as fp:
        label_dict = pickle.load(fp)

    dyn_unc_rank=np.argsort(dynamic_uncertainty)
    keep = []
    for id in range(0, cfg.num_classes):
        samples=set(label_dict[id])
        ordered_samples = [x for x in dyn_unc_rank if x in samples] # samples in 'samples' by the order they appear in 'dyn_unc_rank'
        
        kept_amount = int(len(ordered_samples) * cfg.fraction)
        if kept_amount < cfg.min_samples_per_id:
            if len(ordered_samples) > cfg.min_samples_per_id:
                kept_amount = cfg.min_samples_per_id
            else:
                kept_amount = len(ordered_samples)
        
        keep += list(set(ordered_samples[-kept_amount:]))
    keep=sorted(keep)

    # save final labels to a .py file
    file_content = f"kept_index = {keep}"
    with open(save_path+'/'+cfg.dataset+'_'+str(int(cfg.fraction*100))+'.txt', "w") as file:
        file.write(file_content)

    print(f'Finished selecting {cfg.fraction * 100}% ({len(keep)}) of the {cfg.dataset} dataset ({cfg.num_image})')

if __name__ == "__main__":
    main()