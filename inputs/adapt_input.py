import os
import ujson as json
import utils.inputs.detections_2d as detections_2d
import utils.inputs.detections_3d as detections_3d


#############################################################################
# Combining MOTSFusion segmentations with their corresponding detection info  


def add_detection_info_to_motsfusion_trackrcnn_segmentations(target_seq_name, args):
    """
    Combining MOTSFusion segmentations with their corresponding detection info (track rcnn)

    Args:
        target_seq_name (str): target sequence name

    Returns:
        None
    """
    classes, scores, _, boxes, reids = ([] for _ in range(5))
    detections_2d._load_segmentations_trackrcnn(target_seq_name, classes, scores, _, boxes, reids, motsfusion=True)

    file_path_motsfusion_trackrcnn = os.path.join(os.path.join(
        args.root_dir + "/storage/detections_segmentations_trackrcnn_BB2SegNet/" + args.split[
                                                                                   1:] + "/segmentations_trackrcnn",
        target_seq_name), 'segmentations.json')
    with open(file_path_motsfusion_trackrcnn, 'r+') as file:
        motsfusion_segmentations = json.load(file)
        for frame, masks_frame in enumerate(motsfusion_segmentations):
            for current_seg, class_det, score_det, box_det, reid_det in zip(
                    masks_frame, classes[frame], scores[frame], boxes[frame], reids[frame]):
                score_seg = current_seg['score']
                current_seg['class'] = class_det
                current_seg['box_det'] = box_det
                current_seg['score_det'] = score_det
                current_seg['score'] = score_det
                current_seg['reid'] = reid_det
                current_seg['score_seg'] = score_seg
        file.seek(0)
        json.dump(motsfusion_segmentations, file)
        file.truncate()


def add_detection_info_to_motsfusion_rrc_segmentations(target_seq_name, args):
    """
    Combining MOTSFusion segmentations with their corresponding detection info (rcnn)

    Args:
        target_seq_name (str): target sequence name

    Returns:
        None
    """
    scores, boxes = [], []
    detections_3d._load_detections_motsfison_rrc(target_seq_name, scores, boxes)

    file_path_motsfusion_trackrcnn = os.path.join(os.path.join(
        args.root_dir + "/storage/detections_segmentations_RRC_BB2SegNet/" + args.split[1:] + "/segmentations",
        target_seq_name), 'segmentations.json')
    with open(file_path_motsfusion_trackrcnn, 'r+') as file:
        motsfusion_segmentations = json.load(file)
        for frame, masks_frame in enumerate(motsfusion_segmentations):
            for current_seg, box_det, score_det in zip(
                    masks_frame, boxes[frame], scores[frame]):
                score_seg = current_seg['score']
                current_seg['class'] = 1  # RRC only works for Car
                current_seg['box_det'] = box_det
                current_seg['score_det'] = score_det
                current_seg['score'] = score_det
                current_seg['reid'] = None
                current_seg['score_seg'] = score_seg

        file.seek(0)
        json.dump(motsfusion_segmentations, file)
        file.truncate()
