import numpy as np
import torch
#np.where(np.array(program_lens)==17)
#pr_len_wise_accuracy = {}
#pr_type_wise_accuracy = {}

def old_compute_running_accuracy(labels_argmaxed,
                             logits_argmaxed,
                             accuracy_dict_stats,
                             batch_number,
                            ):
    correct = labels_argmaxed == logits_argmaxed
    correct_mean = correct.float().mean().item()
    #print(correct_mean)
    #input()
    accuracy_dict_stats['all_corrects']+= correct_mean
    accuracy_dict_stats['avg_acc'] = 100 * accuracy_dict_stats['all_corrects'] / (batch_number+1)
    accuracy_dict_stats['num_cases'] += len(labels_argmaxed)
    
    return accuracy_dict_stats
    
    
def update_program_type_and_question_wise_acc(batch_item,
                                             labels_argmaxed,
                                             logits_argmaxed,
                                             accuracy_dict,
                                             batch_number,
                                             ):
    
    if(accuracy_dict is None):
        accuracy_dict = {#'by_question_id':{},#'avg_acc':0.0, 'num_cases':0, 'all_corrects':0},
                         'by_program_type':{},#'avg_acc':0.0, 'num_cases':0, 'all_corrects':0},
                         'by_program_len':{},
                         #'by_program_subtype':{},
                         #'overall': {'avg_acc':0.0, 'num_cases':0, 'all_corrects':0},
                         'overall_total': {'avg_acc':0.0, 'num_cases':0, 'num_correct':0},
                         'overall': {'avg_acc':0.0} #across q types average
                         #'overall_by_question':{'avg_acc':0.0, 'num_cases':0, 'num_correct':0, 'question_tracking':{}},
                        }
    
    b_size = len(batch_item['q_types'])#.s
    for b_i in range(b_size):
        q_type = batch_item['q_types'][b_i].lower()
        pr_len = len(batch_item['programs'][b_i])
        if(q_type not in accuracy_dict['by_program_type']):
            accuracy_dict['by_program_type'][q_type] = {'indv':{'avg_acc':0.0, 'num_cases':0, 'num_correct':0},
                                                        #'by_question':{'avg_acc':0.0, 'num_cases':0, 'num_correct':0,'question_tracking':{}}
                                                       }
        if(pr_len not in accuracy_dict['by_program_len']):
            accuracy_dict['by_program_len'][pr_len] = {'indv':{'avg_acc':0.0, 'num_cases':0, 'num_correct':0},
                                                        #'by_question':{'avg_acc':0.0, 'num_cases':0, 'num_correct':0,'question_tracking':{}}
                                                       }
            
        accuracy_dict['by_program_type'][q_type]['indv']['num_cases']+=1
        accuracy_dict['by_program_len'][pr_len]['indv']['num_cases']+=1
        
        accuracy_dict['overall_total']['num_cases']+=1
        #accuracy_dict['by_program_subtype'][q_subtype]['indv']['num_cases']+=1
        correct = labels_argmaxed[b_i] == logits_argmaxed[b_i]
        
        if(correct):
            accuracy_dict['by_program_type'][q_type]['indv']['num_correct']+=1
            accuracy_dict['by_program_len'][pr_len]['indv']['num_correct']+=1
            accuracy_dict['overall_total']['num_correct']+=1
            
        accuracy_dict['by_program_type'][q_type]['indv']['avg_acc'] = accuracy_dict['by_program_type'][q_type]['indv']['num_correct']/accuracy_dict['by_program_type'][q_type]['indv']['num_cases']
        accuracy_dict['by_program_len'][pr_len]['indv']['avg_acc'] = accuracy_dict['by_program_len'][pr_len]['indv']['num_correct']/accuracy_dict['by_program_len'][pr_len]['indv']['num_cases']
        accuracy_dict['overall_total']['avg_acc'] = accuracy_dict['overall_total']['num_correct']/accuracy_dict['overall_total']['num_cases']
        num_qtypes = len(accuracy_dict['by_program_type'].keys())
        accuracy_dict['overall']['avg_acc'] = sum([accuracy_dict['by_program_type'][qtype]['indv']['avg_acc'] for qtype in accuracy_dict['by_program_type'].keys()]) / num_qtypes
        
    return accuracy_dict