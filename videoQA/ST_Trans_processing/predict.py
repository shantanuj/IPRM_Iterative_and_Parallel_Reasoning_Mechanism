import numpy as np
np.set_printoptions(precision=4)
import copy
import torch
import pickle
import os

from dataloader.action_genome import AG, cuda_collate_fn

from lib.config import Config
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.object_detector import detector
from lib.sttran import STTran
from predict_format_utils import *

conf = Config()
for i in conf.args:
    print(i,':', conf.args[i])

AG_dataset = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                filter_small_box=False if conf.mode == 'predcls' else True)
dataloader = torch.utils.data.DataLoader(AG_dataset, shuffle=False, num_workers=0, collate_fn=cuda_collate_fn)

gpu_device = torch.device('cuda:0')
object_detector = detector(train=False, object_classes=AG_dataset.object_classes, use_SUPPLY=True, mode=conf.mode).to(device=gpu_device)
object_detector.eval()


model = STTran(mode=conf.mode,
               attention_class_num=len(AG_dataset.attention_relationships),
               spatial_class_num=len(AG_dataset.spatial_relationships),
               contact_class_num=len(AG_dataset.contacting_relationships),
               obj_classes=AG_dataset.object_classes,
               enc_layer_num=conf.enc_layer,
               dec_layer_num=conf.dec_layer).to(device=gpu_device)

model.eval()
#ckpt = torch.load("/data/usrdata/shan/ST_Tran_models/sgdet.tar", map_location=gpu_device)
ckpt = torch.load(conf.model_path, map_location=gpu_device)
model.load_state_dict(ckpt['state_dict'], strict=False)
print('*'*50)
print('CKPT {} is loaded'.format(conf.model_path))
#
obj_bboxes_gt = pickle.load(open("/data/usrdata/shan/STAR_VideoQA/obj_bboxes_by_videoid.pkl",'rb'))
person_bboxes_gt = pickle.load(open("/data/usrdata/shan/STAR_VideoQA/person_bboxes_by_videoid.pkl",'rb'))

evaluator1 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint='with')

evaluator2 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint='semi', semithreshold=0.9)

evaluator3 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint='no')
out_pred_path = os.path.join(conf.save_path, "saved_preds.pkl")
out_formatted_person_prediction_path = os.path.join(conf.save_path, "predicted_person_bboxes.pkl")
out_formatted_objs_prediction_path = os.path.join(conf.save_path, "predicted_obj_bboxes_objthresh0.5_relthresh0.6.pkl")

person_bboxes_pred = None
obj_bboxes_pred = None

with torch.no_grad():
    for b, data in enumerate(dataloader):
        print(f'done {b}/{len(dataloader)}')
        
        im_data = copy.deepcopy(data[0].cuda(0))
        im_info = copy.deepcopy(data[1].cuda(0))
        gt_boxes = copy.deepcopy(data[2].cuda(0))
        num_boxes = copy.deepcopy(data[3].cuda(0))
        gt_annotation = AG_dataset.gt_annotations[data[4]]

        entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)

        pred = model(entry)
        #print(dict(pred).keys())
        #input()
        #print(dict(pred))
        #input()
        data_and_pred_tuple = (data, dict(pred))
        pickle.dump(data_and_pred_tuple, open(out_pred_path, 'wb'))
        print(f"saved data and prediction to {out_pred_path}")
        #input()
        
        evaluator1.evaluate_scene_graph(gt_annotation, dict(pred))
        evaluator2.evaluate_scene_graph(gt_annotation, dict(pred))
        evaluator3.evaluate_scene_graph(gt_annotation, dict(pred))
        person_bboxes_pred, obj_bboxes_pred = get_formatted_prediction(st_preds = data_and_pred_tuple,
                         dataloader = dataloader,
                         person_bboxes_pred=person_bboxes_pred, 
                         person_bboxes_gt = person_bboxes_gt, 
                         obj_bboxes_pred = obj_bboxes_pred, 
                         obj_bboxes_gt = obj_bboxes_gt,
                        )
        pickle.dump(person_bboxes_pred, open(out_formatted_person_prediction_path, 'wb'))
        pickle.dump(obj_bboxes_pred, open(out_formatted_objs_prediction_path, 'wb'))
        
        print(f"saved formatted person and obj bbox + relation predictions to {out_formatted_person_prediction_path} and {out_formatted_objs_prediction_path} respectively")
        

print('-------------------------with constraint-------------------------------')
evaluator1.print_stats()
print('-------------------------semi constraint-------------------------------')
evaluator2.print_stats()
print('-------------------------no constraint-------------------------------')
evaluator3.print_stats()
