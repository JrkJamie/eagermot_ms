from __future__ import annotations
from typing import Dict, List
import os
import utils.inputs.utils as utils
import utils.inputs.detections_2d as detections_2d
import utils.inputs.detections_3d as detections_3d
from utils.inputs.bbox import Bbox3d
from utils.inputs.detection_2d import Detection2D


def load_segmentations_trackrcnn(target_seq_name, classes_to_load=None):
    """
    load segmentations of trackrcnn

    Args:
        target_seq_name (str): target sequence name
        classes_to_load (int): class to load

    Returns:
        numpy of classes, scores, masks, boxes, reids
    """
    classes, scores, masks, boxes, reids = ([] for _ in range(5))
    detections_2d._load_segmentations_trackrcnn(
        target_seq_name, classes, scores, masks, boxes, reids, classes_to_load)
    return utils.convert_nested_lists_to_numpy(classes, scores, masks, boxes, reids)


def load_segmentations_motsfusion_rrc(target_seq_name, args, classes_to_load=None):
    """
    load motsfusion segmentations of rrc

    Args:
        target_seq_name (str): target sequence name
        classes_to_load (int): class to load

    Returns:
        numpy of classes, scores, masks, boxes, reids
    """
    classes, scores, masks, boxes, reids = ([] for _ in range(5))
    detections_2d._load_segmentations_motsfusion(os.path.join(
        args.root_dir + "/storage/detections_segmentations_RRC_BB2SegNet/" + args.split[1:] + "/segmentations",
        target_seq_name), target_seq_name,
        classes, scores, masks, boxes, reids, classes_to_load)
    return utils.convert_nested_lists_to_numpy(classes, scores, masks, boxes, reids)


def load_segmentations_motsfusion_trackrcnn(target_seq_name, args, classes_to_load=None):
    """
    load motsfusion segmentations of trackrcnn

    Args:
        target_seq_name (str): target sequence name
        classes_to_load (int): class to load

    Returns:
        numpy of classes, scores, masks, boxes, reids
    """
    classes, scores, masks, boxes, reids = ([] for _ in range(5))
    detections_2d._load_segmentations_motsfusion(os.path.join(
        args.root_dir + "/storage/detections_segmentations_trackrcnn_BB2SegNet/" + args.split[
                                                                                   1:] + "/segmentations_trackrcnn",
        target_seq_name), target_seq_name,
        classes, scores, masks, boxes, reids, classes_to_load)
    return utils.convert_nested_lists_to_numpy(classes, scores, masks, boxes, reids)


def load_segmentations_motsfusion_best(target_seq_name, args, classes_to_load=None):
    """
    load best motsfusion segmentations of trackrcnn

    Args:
        target_seq_name (str): target sequence name
        classes_to_load (int): class to load

    Returns:
        numpy of classes, scores, masks, boxes, reids
    """
    classes, scores, masks, boxes, reids = ([] for _ in range(5))
    detections_2d._load_segmentations_motsfusion(os.path.join(
        args.root_dir + "/storage/detections_segmentations_RRC_BB2SegNet/" + args.split[1:] + "/segmentations",
        target_seq_name), target_seq_name,
        classes, scores, masks, boxes, reids, [detections_2d.CAR_CLASS])
    # print('loaded MOTSFusion', len(classes), len(classes[0]))
    detections_2d._load_segmentations_motsfusion(os.path.join(
        args.root_dir + "/storage/detections_segmentations_trackrcnn_BB2SegNet/" + args.split[
                                                                                   1:] + "/segmentations_trackrcnn",
        target_seq_name), target_seq_name,
        classes, scores, masks, boxes, reids, [detections_2d.PED_CLASS])
    # print('loaded Both', len(classes), len(classes[0]), len(masks), len(masks[0]))
    # print(masks[0])

    return utils.convert_nested_lists_to_numpy(classes, scores, masks, boxes, reids)


def load_segmentations_tracking_best(target_seq_name, args, classes_to_load=None):
    """
    Load 2D detections for "car" given by MOTSFusion and for "ped" given by TrackRCNN

    Args:
        target_seq_name (str): target sequence name
        classes_to_load (int): class to load

    Returns:
        numpy of classes, scores, masks, boxes, reids
    """
    classes, scores, masks, boxes, reids = ([] for _ in range(5))
    detections_2d._load_segmentations_motsfusion(os.path.join(
        args.root_dir + "/storage/detections_segmentations_RRC_BB2SegNet/" + args.split[1:] + "/segmentations",
        target_seq_name), target_seq_name,
        classes, scores, masks, boxes, reids, args, [detections_2d.CAR_CLASS])
    detections_2d._load_segmentations_trackrcnn(
        target_seq_name, classes, scores, masks, boxes, reids, args, [detections_2d.PED_CLASS])
    return utils.convert_nested_lists_to_numpy(classes, scores, masks, boxes, reids)


def load_detections_3d(dets_3d_source: str, seq_name: str, param, args) -> Dict[str, List[Bbox3d]]:
    """
    high API to load 3d detections

    Args:
        dets_3d_source (str): detection method
        seq_name (str): sequence name

    Returns:
        dict(bbox3d)
    """
    if dets_3d_source == "ab3dmot":
        return detections_3d._load_detections_ab3dmot(seq_name, param, args)
    if dets_3d_source == "pointgnn_t3":
        return detections_3d._load_detections_pointgnn(
            [os.path.join(args.root_dir + "/storage/pointgnn/" + args.split[1:], split, seq_name, 'data') for split in
             param['_SPLITS_POINTGNN_T3']], param, args)
    if dets_3d_source == "pointgnn_t2":
        return detections_3d._load_detections_pointgnn(
            [os.path.join(args.root_dir + "/storage/pointgnn/" + args.split[1:], split, seq_name, 'data') for split in
             param['_SPLITS_POINTGNN_T2']], param, args)
    if dets_3d_source == "3dop":
        return detections_3d.load_detections_3dop(seq_name, param, args)
    if dets_3d_source == "center_point":
        return detections_3d.load_detections_centerpoint(param, args)
    raise NotImplementedError


def load_detections_2d_kitti(dets_2d_source: str, seq_name: str, param, args):
    """
    high API to load 2d detections in kitti

    Args:
        dets_3d_source (str): detection method
        seq_name (str): sequence name

    Returns:
        dict(bbox3d)
    """
    if dets_2d_source == param['TRACKRCNN']:
        return load_segmentations_trackrcnn(seq_name, args)
    elif dets_2d_source == param['MOTSFUSION_RRC']:
        return load_segmentations_motsfusion_rrc(seq_name, args)
    elif dets_2d_source == param['MOTSFUSION_TRACKRCNN']:
        return load_segmentations_motsfusion_trackrcnn(seq_name, args)
    elif dets_2d_source == param['MOTSFUSION_BEST']:
        return load_segmentations_motsfusion_best(seq_name, args)
    elif dets_2d_source == param['TRACKING_BEST']:
        return load_segmentations_tracking_best(seq_name, args)
    raise NotImplementedError


def load_detections_2d_kitti_new(dets_2d_source: str, seq_name: str, param, args) -> Dict[
    str, Dict[str, List[Detection2D]]]:
    """
    high API to load 2d detections in kitti new, Should return a dict mapping frame to each camera with its detections

    Args:
        dets_3d_source (str): detection method
        seq_name (str): sequence name

    Returns:
        dict(bbox3d)
    """
    if dets_2d_source == param['MMDETECTION_CASCADE_NUIMAGES']:
        return detections_2d.load_detections_2d_mmdetection_kitti(seq_name, args)
    raise NotImplementedError


def load_detections_2d_nuscenes(dets_2d_source: str, seq_name: str, param, args) -> Dict[
    str, Dict[str, List[Detection2D]]]:
    """
    high API to load 2d detections in nuscenes

    Args:
        dets_3d_source (str): detection method
        seq_name (str): sequence name

    Returns:
        dict(bbox3d)
    """
    """ Should return a dict mapping frame to each camera with its detections """
    if dets_2d_source == param['EFFICIENT_DET']:
        return detections_2d.load_detections_2d_efficient_det(seq_name, param, args)
    if dets_2d_source == param['MMDETECTION_CASCADE_NUIMAGES']:
        return detections_2d.load_detections_2d_mmdetection_nuscenes(seq_name, param, args)
    raise NotImplementedError


def load_annotations_kitti(seq_name: str) -> Dict[str, List[Bbox3d]]:
    """
    high API to load 2d annotations in kitti

    Args:
        dets_3d_source (str): detection method
        seq_name (str): sequence name

    Returns:
        dict(bbox3d)
    """
    return detections_3d.load_annotations_kitti(seq_name)
