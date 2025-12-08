import numpy as np
import torch
#np.where(np.array(program_lens)==17)
#pr_len_wise_accuracy = {}
#pr_type_wise_accuracy = {}

def compute_running_accuracy(labels_argmaxed,
                             logits_argmaxed,
                             accuracy_dict_stats,
                             batch_i,
                            ):
    correct = labels_argmaxed == logits_argmaxed
    correct_mean = correct.float().mean().item()
    #print(correct, correct.float().sum()) 
    #input()
    #print(correct_mean)
    #input()
    accuracy_dict_stats['all_corrects']+= correct_mean
    accuracy_dict_stats['avg_acc_batch'] = 100 * accuracy_dict_stats['all_corrects'] / (batch_i+1)
    accuracy_dict_stats['num_cases'] += len(labels_argmaxed)
    accuracy_dict_stats['num_correct'] += int(correct.float().sum())
    accuracy_dict_stats['avg_acc_metric'] = accuracy_dict_stats['num_correct'] / accuracy_dict_stats['num_cases']
    return accuracy_dict_stats
    
    
def update_program_type_and_len_wise_acc(batch_programs, 
                                         batch_program_types,
                                         labels_argmaxed,
                                         logits_argmaxed,
                                         accuracy_dict,
                                         batch_i,
                                         program_lens_dict,
                                         program_types_dict,
                                        ):
    
    if(accuracy_dict is None):
        accuracy_dict = {'by_program_len':{},#'avg_acc':0.0, 'num_cases':0, 'all_corrects':0},
                         'by_program_type':{},#'avg_acc':0.0, 'num_cases':0, 'all_corrects':0},
                         'overall': {'avg_acc_batch':0.0, 'num_cases':0, 'all_corrects':0, 'avg_acc_metric':0.0, 'num_correct':0}
                        }
    
    accuracy_dict['overall'] = compute_running_accuracy(labels_argmaxed, 
                                                        logits_argmaxed,
                                                        accuracy_dict['overall'],
                                                        batch_i,
                                                       )
    
    
    batch_program_lens = np.array([len(prg) for prg in batch_programs])
    for pr_len in program_lens_dict:
        if(pr_len not in accuracy_dict['by_program_len']):
            accuracy_dict['by_program_len'][pr_len] = {'avg_acc_batch':0.0, 'num_cases':0, 'all_corrects':0, 'avg_acc_metric':0.0, 'num_correct':0}
                                                       
        indices = np.where(batch_program_lens==pr_len)
        if(not indices[0].shape[0]>0):
            continue
        #print(indices[0].shape[0]>0, indices[0].shape[0] )
        #input()
        accuracy_dict['by_program_len'][pr_len] = compute_running_accuracy(labels_argmaxed[indices],
                                                                           logits_argmaxed[indices],
                                                                           accuracy_dict['by_program_len'][pr_len],
                                                                           batch_i,
                                                                       )
    
    batch_program_types = np.array(batch_program_types)
    for pr_type in program_types_dict:
        if(pr_type not in accuracy_dict['by_program_type']):
            accuracy_dict['by_program_type'][pr_type] = {'avg_acc_batch':0.0, 'num_cases':0, 'all_corrects':0, 'avg_acc_metric':0.0, 'num_correct':0}
        
        indices = np.where(batch_program_types==pr_type)
        if(not indices[0].shape[0]>0):
            continue
        accuracy_dict['by_program_type'][pr_type] = compute_running_accuracy(labels_argmaxed[indices],
                                                                            logits_argmaxed[indices],
                                                                            accuracy_dict['by_program_type'][pr_type],
                                                                            batch_i,
                                                                            )
    
    return accuracy_dict