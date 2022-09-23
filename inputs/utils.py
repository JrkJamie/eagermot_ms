import os
import ujson as json
from typing import Mapping, Dict
import numpy as np
import dataset.base_class.reporting as reporting_nuscenes

AB3DMOT = 'ab3dmot'
POINTGNN_T3 = 'pointgnn_t3'
POINTGNN_T2 = 'pointgnn_t2'
_SPLITS_POINTGNN_T3 = ['results_tracking_car_auto_t3_trainval', 'results_tracking_ped_cyl_auto_trainval']
_SPLITS_POINTGNN_T2 = ['results_tracking_car_auto_t2_train', 'results_tracking_ped_cyl_auto_trainval']
CENTER_POINT = 'center_point'
EFFICIENT_DET = 'efficient_det'
STEREO_3DOP = '3dop'
TRACKRCNN = 'trackrcnn'
MOTSFUSION_RRC = 'motsfusion_rrc'
MOTSFUSION_TRACKRCNN = 'motsfusion_trackrcnn'
MOTSFUSION_BEST = 'motsfusion_best'
TRACKING_BEST = 'rrc_trackrcnn'
MMDETECTION_CASCADE_NUIMAGES = 'mmdetection_cascade_nuimages'
_SPLITS_AB3DMOT = ['car_3d_det', 'ped_3d_det']


def parse_and_add_seg_for_frame(frame, current_seg, classes_to_load, parse_func,
                                classes, scores, masks, boxes, reids):
    cur_class, score, mask, box, reid = parse_func(current_seg)
    if classes_to_load is None or cur_class in classes_to_load:
        classes[frame].append(cur_class)
        scores[frame].append(score)
        masks[frame].append(mask)
        boxes[frame].append(box)
        reids[frame].append(reid)


def pad_lists_if_necessary(frame_num, collection_of_lists):
    while len(collection_of_lists[0]) < frame_num + 1:
        for lst in collection_of_lists:
            lst.append([])


def convert_nested_lists_to_numpy(classes, scores, masks, boxes, reids):
    for t in range(len(classes)):
        classes[t] = np.array(classes[t])
        scores[t] = np.array(scores[t])
        masks[t] = np.array(masks[t])
        boxes[t] = np.vstack(boxes[t]) if len(classes[t]) > 0 else np.array([])
        reids[t] = np.array(reids[t])
    return boxes, scores, reids, classes, masks


mmdetection_nuimages_class_mapping_nuscenes: Mapping[str, int] = {
    "0": reporting_nuscenes.id_from_name("pedestrian"),
    "3": reporting_nuscenes.id_from_name("bicycle"),
    "4": reporting_nuscenes.id_from_name("bus"),
    "5": reporting_nuscenes.id_from_name("car"),
    "7": reporting_nuscenes.id_from_name("motorcycle"),
    "8": reporting_nuscenes.id_from_name("trailer"),
    "9": reporting_nuscenes.id_from_name("truck"),
}


def load_json_for_sequence(folder_path: str, target_seq_name: str) -> Dict:
    folder_dir = os.path.join(folder_path)
    print(folder_dir)
    assert os.path.isdir(folder_dir)

    # Parse sequences
    filepath = None
    for scene_json in os.listdir(folder_dir):
        if scene_json.startswith(target_seq_name):
            filepath = os.path.join(folder_dir, scene_json)
            break
    else:
        raise NotADirectoryError(f"No detections for {target_seq_name}")

    print(f"Parsing {filepath}")
    with open(filepath, 'r') as f:
        all_detections = json.load(f)
    return all_detections
