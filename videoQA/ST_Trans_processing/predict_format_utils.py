import torch
import numpy as np

#person_bboxes_pred = {}
#obj_bboxes_pred = {}


def get_formatted_prediction(st_preds,
                             dataloader,
                             person_bboxes_pred, 
                             person_bboxes_gt, 
                             obj_bboxes_pred, 
                             obj_bboxes_gt,
                             obj_score_thresh = 0.5,
                             rel_att_thresh = 0.6,
                             rel_spatial_thresh = 0.6,
                             rel_contact_thresh = 0.6,
                            ):
    if(person_bboxes_pred is None):
        person_bboxes_pred = {}
    if(obj_bboxes_pred is None):
        obj_bboxes_pred = {}
        
        
    ##Person bboxes storage
    if or_video_id in person_bboxes_pred:
        print(f"ERROR: video id {or_video_id} already present in person_bboxes_pred dictionary")
        #continue
    if(or_video_id not in person_bboxes_gt):
        print(f"ERROR: video id {or_video_id} not found in person_bboxes_gt dictionary")
        #continue
    person_bboxes_pred[or_video_id] = {'frames':{}, 'vid_info':person_bboxes_gt[or_video_id]}

    for frame_i, vid_frame_id in enumerate(or_video_id_frames):
        frame_id = vid_frame_id.split('/')[-1]
        if(frame_id not in person_bboxes_gt[or_video_id]['frames']):
            print(f"ERROR: frame id {frame_id} not found in person_bboxes_gt dictionary")
            continue

        person_idx = st_preds['human_idx'][frame_i]
        bbox_coords = st_preds['boxes'][person_idx][0][1:].unsqueeze(0).cpu().clone().numpy()
        bbox_score = st_preds['pred_scores'][person_idx].cpu().clone().numpy()
        bbox_label = dataloader.dataset.object_classes[st_preds['pred_labels'][person_idx]]
        if(bbox_label not in ['person']):
            print(f"ERROR: for frame id {frame_id}, detected bbox_label for person is {bbox_label} and not 'person'")
        #print(bbox_coords.shape)
        person_bboxes_pred[or_video_id]['frames'][frame_id] = {'bbox': bbox_coords, 
                                                  'bbox_score': bbox_score,
                                                  'bbox_obj_label':bbox_label,
                                                  'bbox_size': person_bboxes_gt[or_video_id]['frames'][frame_id]['bbox_size'],
                                                  'keypoints': person_bboxes_gt[or_video_id]['frames'][frame_id]['keypoints'],
                                                  'keypoints_logits': person_bboxes_gt[or_video_id]['frames'][frame_id]['keypoints_logits'],

                                                 }


    ##Obj bboxes storage
    if or_video_id in obj_bboxes_pred:
        print(f"ERROR: video id {or_video_id} already present in obj_bboxes_pred dictionary")
        #continue
    if(or_video_id not in obj_bboxes_gt):
        print(f"ERROR: video id {or_video_id} not found in obj_bboxes_gt dictionary")
        #continue
    obj_bboxes_pred[or_video_id] = {'frames':{}, 'vid_info':obj_bboxes_gt[or_video_id]}
    assert(st_preds['im_idx'].size(0)==st_preds['pair_idx'].size(0))

    image_idx_ints = set([int(im_id) for im_id in list(st_preds['im_idx'].cpu().clone().numpy())])
    for frame_i in image_idx_ints: #
        frame_i = int(frame_i)
        or_vid_frame_id = or_video_id_frames[frame_i]
        or_frame_id = or_vid_frame_id.split('/')[-1]
        #print(or_frame_id, obj_bboxes_pred[or_video_id]['frames'].keys())
        #input()
        if(or_frame_id not in obj_bboxes_gt[or_video_id]['frames']):
            print(f"WARNING: frame id {frame_id} not found in obj_bboxes_gt dictionary")

        if(or_frame_id not in obj_bboxes_pred[or_video_id]['frames']):
            obj_bboxes_pred[or_video_id]['frames'][or_frame_id] = []
        #else:
         #   print(f"Warning: frame id {or_frame_id} already found in obj_bboxes_pred dictionary")

        objs_id_for_frame = torch.where(st_preds['boxes'][:,0]==frame_i)[0]
        if(len(objs_id_for_frame)<=0):
            print(f"WARNING: found 0 objs for original frame id {frame_id} (original frame id:)")
        human_obj_idx = int(st_preds['human_idx'][frame_i][0])
        #print('obj_ids', objs_id_for_frame)
        #input()
        obj_labels_covered_with_conf_and_listposition = {} #maintain objects already instantiated for the frame
        for idx in objs_id_for_frame:
            #print('hello', idx)
            #input()
            idx=int(idx)
            #idx_int = int(idx)
            if(idx==human_obj_idx):
                print(f"ignoring obj id {idx} as it is human index")
                continue
            #print('ABC', idx)

            obj_bbox = st_preds['boxes'][idx].cpu().clone().numpy()[1:]#[1:]
            obj_bbox[2] = obj_bbox[2] - obj_bbox[0]
            obj_bbox[3] = obj_bbox[3] - obj_bbox[1]
            obj_bbox_score = st_preds['pred_scores'][idx].unsqueeze(-1).cpu().clone().numpy()
            obj_bbox_label = dataloader.dataset.object_classes[st_preds['pred_labels'][idx]]
            
            if(obj_bbox_score[0]<obj_score_thresh):
                print(f"WARNING: Skipping obj {obj_bbox_label} with bbox score {obj_bbox_score[0]} as < thresh {obj_score_thresh}")
                continue

            if(obj_bbox_label in obj_labels_covered_with_conf_and_listposition):
                already_existing_obj_score_and_listposition = obj_labels_covered_with_conf_and_listposition[obj_bbox_label]
                if(float(obj_bbox_score[0])> already_existing_obj_score_and_listposition[0]):
                    print(f"WARNING: Replacing existing obj label {obj_bbox_label} with lower bbox score {already_existing_obj_score_and_listposition[0]} as new one has higher score {obj_bbox_score[0]}")
                    obj_bboxes_pred[or_video_id]['frames'][or_frame_id].pop(already_existing_obj_score_and_listposition[1])
                    obj_labels_covered_with_conf_and_listposition[obj_bbox_label] = (float(obj_bbox_score[0]), len(obj_bboxes_pred[or_video_id]['frames'][or_frame_id]))
                else:
                    print(f"WARNING: Skipping obj {obj_bbox_label} with bbox score {obj_bbox_score[0]} as < already existing obj label {obj_bbox_label} with score {already_existing_obj_score_and_listposition[0]}")
                    continue
            else:
                obj_labels_covered_with_conf_and_listposition[obj_bbox_label] = (float(obj_bbox_score[0]), len(obj_bboxes_pred[or_video_id]['frames'][or_frame_id]))
            obj_struct = {'class': obj_bbox_label,
                          'obj_score': obj_bbox_score,
                           'bbox': tuple(obj_bbox),
                           'attention_relationship': [],
                           'attention_relationship_scores': [],
                           'spatial_relationship': [],
                          'spatial_relationship_scores': [],
                           'contacting_relationship': [],
                          'contacting_relationship_scores': [],
                          'metadata': None,
                           #'metadata': {'tag': '1X0M3.mp4/groceries/000523', 'set': 'train'},
                           'visible': True}
            rels_ids = torch.where(st_preds['pair_idx'][:,1]==idx)
            att_rel_distrb = st_preds['attention_distribution'][rels_ids][0]

            for rel_i, rel_score in enumerate(att_rel_distrb):
                if(rel_score>=rel_att_thresh):
                    rel_name = dataloader.dataset.attention_relationships[rel_i]
                    obj_struct['attention_relationship'].append(rel_name)
                    obj_struct['attention_relationship_scores'].append(float(rel_score))

            spatial_rel_distrb = st_preds['spatial_distribution'][rels_ids][0]
            for rel_i, rel_score in enumerate(spatial_rel_distrb):
                if(rel_score>=rel_spatial_thresh):
                    rel_name = dataloader.dataset.spatial_relationships[rel_i]
                    obj_struct['spatial_relationship'].append(rel_name)
                    obj_struct['spatial_relationship_scores'].append(float(rel_score))
            contact_rel_distrb = st_preds['contacting_distribution'][rels_ids][0]
            for rel_i, rel_score in enumerate(contact_rel_distrb):
                if(rel_score>=rel_contact_thresh):
                    rel_name = dataloader.dataset.contacting_relationships[rel_i]
                    obj_struct['contacting_relationship'].append(rel_name)
                    obj_struct['contacting_relationship_scores'].append(float(rel_score))

            obj_bboxes_pred[or_video_id]['frames'][or_frame_id].append(obj_struct)
            
    return person_bboxes_pred, obj_bboxes_pred