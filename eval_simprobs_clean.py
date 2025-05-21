import os
import torch
import numpy as np
import re
import pickle

import torch.distributed as dist

from config.config import config as cfg

def get_average_prediction(num_image, num_epoch):
    pred_prob_his = torch.zeros(num_image, num_epoch).double()

    # sorting the file list by epoch (ascending order)
    pred_prob_history_files = os.listdir(f'{cfg.coreset_input}/pred_prob_clean')
    pred_prob_history_files.sort(key=lambda x: int(re.findall("\d+", x)[0]))

    for (idx, pred_prob_file) in enumerate(pred_prob_history_files):
        pred_prob_his[:,idx] = torch.from_numpy(np.loadtxt(f'{cfg.coreset_input}/pred_prob_clean/{pred_prob_file}'))

    avg_prediction = np.squeeze(np.mean(np.array(pred_prob_his), axis=1))

    avg_pred_rank = np.argsort(avg_prediction)

    avg_prediction = torch.from_numpy(avg_prediction)
    avg_prediction = avg_prediction.view(avg_prediction.shape[0], 1)
    
    return avg_prediction, avg_pred_rank

def main():
    dist.init_process_group(backend='nccl', init_method='env://')
    np.random.seed(73)

    if cfg.use_all_epochs:
        save_path = cfg.coreset_input + '/original/' + cfg.original_net_loss + '/coresets_eval_simprobs_clean/' + cfg.coreset_order + '/all'
        pred_prob_his, avg_pred_rank = get_average_prediction(cfg.num_image, cfg.num_epoch)
    else:
        save_path = cfg.coreset_input + '/original/' + cfg.original_net_loss + '/coresets_eval_simprobs_clean/' + cfg.coreset_order + '/epoch_' + str(cfg.eval_epoch)
        pred_prob_his = torch.zeros(cfg.num_image, 1).double()
        pred_prob_his[:, 0] = torch.from_numpy(np.loadtxt(f'{cfg.coreset_input}/original/{cfg.original_net_loss}/pred_prob_clean/prob_{cfg.eval_epoch}.txt'))
        avg_pred_rank = np.argsort(np.squeeze(np.array(pred_prob_his)))

    num_negatives = (pred_prob_his < 0).sum().item()
    print(f'Number of negative values in pred_prob_his: {num_negatives}')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open('/data/mcaldeir/output/labels/label_dict.pkl', 'rb') as fp:
        label_dict = pickle.load(fp)

    keep = []
    for id in range(0, cfg.num_classes):
        samples = set(label_dict[id])
        ordered_samples = [x for x in avg_pred_rank if x in samples and pred_prob_his[x] >= 0] # samples in 'samples' by the order they appear in 'dyn_unc_rank' and excluding the samples removed after cleaning

        # variable initialization
        threshold_decrease = 0
    
        if len(ordered_samples) > cfg.min_samples_per_id:
            final_samples = [ordered_samples[-1]]
            while len(final_samples) < cfg.min_samples_per_id:
                
                min_kept_avg = pred_prob_his[ordered_samples[-1]]
                final_samples = [ordered_samples[-1]]
                for sample in range(len(ordered_samples)-2, -1, -1):
                    if ((min_kept_avg[0] - pred_prob_his[ordered_samples[sample]][0])).item() > (1 + threshold_decrease) * cfg.threshold:
                        final_samples.append(ordered_samples[sample])
                        min_kept_avg = pred_prob_his[ordered_samples[sample]]
                threshold_decrease -= 0.01 # decreases the threshold by 1% in case it is too big for the current id to keep at least 'cfg.min_samples_per_id' samples
        else:
            final_samples = ordered_samples
        keep += list(set(final_samples))
    keep=sorted(keep)

    # save final labels to a .py file
    file_content = f"kept_index = {keep}"
    with open(save_path+'/'+cfg.dataset+'_'+str(int(cfg.threshold*10000))+'.txt', "w") as file:
        file.write(file_content)

    if dist.get_rank() == 0:
        print(f'Finished selecting images from the {cfg.dataset} dataset using the threshold {cfg.threshold}. Images left: {len(keep)}')

if __name__ == "__main__":
    main()