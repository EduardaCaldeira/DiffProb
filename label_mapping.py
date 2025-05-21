import os
import torch
import re
import numpy as np
import ast
import pickle

from config.config import config as cfg

def main():
    path = '../../../../data/mcaldeir/data_pruning/original/' + cfg.original_net_loss + '/coresets_' + cfg.coreset_method + '/' + cfg.coreset_order
    if cfg.coreset_method == 'eval_simprobs' or cfg.coreset_method=='eval_simprobs_clean':
        file_path = path + '/epoch_' + str(cfg.eval_epoch) + '/' + cfg.dataset + '_' + str(int(cfg.threshold * 10000)) + '.txt'
    else:
        file_path = path + '/' + cfg.dataset + '_' + str(int(cfg.fraction * 100)) + '.txt'

    with open('../../../../data/mcaldeir/output/labels/label_dict.pkl', 'rb') as fp:
        label_dict = pickle.load(fp)

    with open(file_path, "r") as file:
        content = file.read()
    parsed_content = ast.literal_eval(content.split('=', 1)[1].strip())

    total_ids=0
    label_map = {}
    for id in range(0, len(label_dict)):
        for sample in range(0, len(label_dict[id])):
            if label_dict[id][sample] in parsed_content:
                if cfg.coreset_method=='eval_simprobs_clean':
                    label_map[id]=total_ids
                total_ids+=1
                break
    
    if cfg.coreset_method=='eval_simprobs_clean':
        with open(path+'/label_map_'+str(int(cfg.threshold*10000))+'.pkl', 'wb') as fp:
            pickle.dump(label_map, fp)
            print('Label map saved successfully!')

    if cfg.coreset_method=='eval_simprobs_clean' or cfg.coreset_method=='eval_simprobs':
        print(f'The total number of considered ids ({cfg.coreset_method}, {cfg.coreset_order}, {cfg.threshold}) is {total_ids}')
    else:
        print(f'The total number of considered ids ({cfg.coreset_method}, {cfg.coreset_order}, {int(cfg.fraction*100)}) is {total_ids}')

if __name__ == "__main__":
    main()