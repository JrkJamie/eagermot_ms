from __future__ import annotations
import os
import ujson as json
from typing import Dict, List
from collections import defaultdict

NO_LABEL = -1
CAR_CLASS = 1
PED_CLASS = 2
DET_TO_TRACK_SEG_CLASS = {1: ('Pedestrian', PED_CLASS), 2: ('Car', CAR_CLASS), 3: ('Cyclist', -1)}
SEG_TO_TRACK_CLASS = {CAR_CLASS: 'Car', PED_CLASS: 'Pedestrian', 3: 'Cyclist'}

import utils.inputs.utils as utils
import utils.inputs.detection_2d as detection_2d
import utils.inputs.bbox as inputs_bbox


# credit to TrackR-CNN https://github.com/VisualComputingInstitute/TrackR-CNN
def _load_segmentations_trackrcnn(target_seq_name,
                                  classes, scores, masks, boxes, reids, args, classes_to_load=None, motsfusion=False):
    if motsfusion:
        with open(os.path.join(args.root_dir + "/storage/detections_segmentations_trackrcnn_BB2SegNet/" + args.split[
                                                                                                          1:] + "/detections_segmentations_trackrcnn",
                               '%s.txt' % target_seq_name)) as f:
            content = f.readlines()
    else:
        os.path.join(args.root_dir + "/storage/trackrcnn/" + args.split[1:], '%s.txt' % target_seq_name)
        with open(
                os.path.join(args.root_dir + "/storage/trackrcnn/" + args.split[1:], '%s.txt' % target_seq_name)) as f:
            content = f.readlines()

    for line in content:
        entries = line.split(' ')
        frame = int(entries[0])
        utils.pad_lists_if_necessary(frame, (classes, scores, masks, boxes, reids))
        utils.parse_and_add_seg_for_frame(frame, entries, classes_to_load, parse_trackrcnn_seg,
                                          classes, scores, masks, boxes, reids)


def _load_segmentations_motsfusion(motsfusion_dir, target_seq_name,
                                   classes, scores, masks, boxes, reids, args, classes_to_load=None):
    with open(os.path.join(motsfusion_dir, 'segmentations.json'), 'r') as file:
        sequence_json = json.load(file)

    for frame, frame_masks in enumerate(sequence_json):
        utils.pad_lists_if_necessary(frame, (classes, scores, masks, boxes, reids))
        for seg in frame_masks:
            utils.parse_and_add_seg_for_frame(frame, seg, classes_to_load, parse_motsfusion_seg,
                                              classes, scores, masks, boxes, reids)


def parse_trackrcnn_seg(seg_values):
    """
    Returns class, score, mask, bbox and reid parsed from input
    
    Args:
        seg_values (numpy): seg values

    Returns:
        class, score, mask, bbox and reid
    """
    mask = {'size': [int(seg_values[7]), int(seg_values[8])],
            'counts': seg_values[9].strip().encode(encoding='UTF-8')}
    box = (int(float(seg_values[1])), int(float(seg_values[2])),
           int(float(seg_values[3])), int(float(seg_values[4])))
    return (int(seg_values[6]), float(seg_values[5]), mask, box, [float(e) for e in seg_values[10:]])


def parse_motsfusion_seg(seg_json):
    """ 
    Returns class, score, mask, bbox and reid parsed from input 
    If throws error, need to run methods in adapt_input.py to adapt MOTSFusion segmentations - 
    to have a class field taken from the corresponding TrackRCNN files, see functions below:
    * add_detection_info_to_motsfusion_trackrcnn_segmentations()
    * add_detection_info_to_motsfusion_rrc_segmentations()

    Args:
        seg_json (numpy): seg values in json

    Returns:
        class, score, mask, bbox and reid
    """
    mask = {'size': seg_json['size'],
            'counts': seg_json['counts'].strip().encode(encoding='UTF-8')}
    return (int(seg_json['class']), float(seg_json['score']),
            mask, seg_json['box_det'], None)  # No ReID for MOTSFusion


def load_detections_2d_efficient_det(seq_name: str, param, args) -> Dict[
    str, Dict[str, List[detection_2d.Detection2D]]]:
    """
    load 2d detection from efficient

    Args:
        seq_name (str): sequence name

    Returns:
        dict(Detection2D)
    """
    all_dets = utils.load_json_for_sequence(args.root_dir + "/storage/efficientdet/" + args.split[1:], seq_name)
    frames_cam_tokens_detections: Dict[str, Dict[str, List[detection_2d.Detection2D]]
    ] = defaultdict(lambda: defaultdict(list))

    # frame_token: cam_data_token: [ [image_id, ymin, xmin, ymax, xmax, score, class] ]
    for frame_token, cam_detections in all_dets.items():
        for cam_data_token, detections in cam_detections.items():
            for detection_data in detections:
                class_id = utils.coco_class_id_mapping.get(int(detection_data[6]), None)
                if class_id is None:
                    continue

                bbox = inputs_bbox.Bbox2d(detection_data[2], detection_data[1], detection_data[4], detection_data[3])
                score = float(detection_data[5])
                detection = detection_2d.Detection2D(bbox, "", score, class_id)
                frames_cam_tokens_detections[frame_token][cam_data_token].append(detection)
    return frames_cam_tokens_detections


def load_detections_2d_mmdetection_nuscenes(seq_name: str, param, args) -> Dict[
    str, Dict[str, List[detection_2d.Detection2D]]]:
    """
    load 2d detection from mmdetection in nuscenes

    Args:
        seq_name (str): sequence name

    Returns:
        dict(Detection2D)
    """
    all_dets = utils.load_json_for_sequence(args.root_dir + "/storage/mmdetection_cascade_x101/" + args.split[1:],
                                            seq_name)
    frames_cam_tokens_detections: Dict[str, Dict[str, List[detection_2d.Detection2D]]
    ] = defaultdict(lambda: defaultdict(list))

    # frame_token: cam_data_token: class_label: [ [xmin, ymin, xmax, ymax, score] ]
    for frame_token, cam_detections in all_dets.items():
        for cam_data_token, class_detections in cam_detections.items():
            for class_label, detections_list in class_detections.items():
                for detection_data in detections_list:
                    class_id = utils.mmdetection_nuimages_class_mapping_nuscenes.get(class_label, None)
                    if class_id is None:  # non-tracking class
                        continue

                    bbox = inputs_bbox.Bbox2d(detection_data[0], detection_data[1], detection_data[2],
                                              detection_data[3])
                    score = float(detection_data[4])
                    detection = detection_2d.Detection2D(bbox, "", score, class_id)
                    frames_cam_tokens_detections[frame_token][cam_data_token].append(detection)
    return frames_cam_tokens_detections


def load_detections_2d_mmdetection_kitti(seq_name: str, args) -> Dict[str, Dict[str, List[detection_2d.Detection2D]]]:
    """
    load 2d detection from mmdetection in kitti

    Args:
        seq_name (str): sequence name

    Returns:
        dict(Detection2D)
    """
    all_dets = utils.load_json_for_sequence(args.root_dir + "/storage/mmdetection_cascade_x101_kitti/" + args.split[1:],
                                            seq_name)
    cam = "image_02"
    frames_cam_detections: Dict[str, Dict[str, List[detection_2d.Detection2D]]
    ] = defaultdict(lambda: defaultdict(list))

    # frame_token: cam_data_token: class_label: [ [xmin, ymin, xmax, ymax, score] ]
    for frame_token, class_detections in all_dets.items():
        for class_label, detections_list in class_detections.items():
            for detection_data in detections_list:
                class_id = utils.mmdetection_nuimages_class_mapping_kitti.get(class_label, None)
                if class_id is None:  # non-tracking class
                    continue

                bbox = inputs_bbox.Bbox2d(detection_data[0], detection_data[1], detection_data[2], detection_data[3])
                score = float(detection_data[4])
                detection = detection_2d.Detection2D(bbox, cam, score, class_id)
                frames_cam_detections[frame_token][cam].append(detection)
    return frames_cam_detections
