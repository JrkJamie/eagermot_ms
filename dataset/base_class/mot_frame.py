from __future__ import annotations
import os
import time, pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Iterable, Tuple, Dict, Mapping, Set
from collections import defaultdict
import numpy as np
import dataset.base_class.mot_sequence as mot_sequence
from utils.inputs.bbox import Bbox3d, Bbox2d
from utils.inputs.detection_2d import Detection2D
from utils.inputs.detections_2d import SEG_TO_TRACK_CLASS
from utils.fused_instance import Source, FusedInstance
from utils.tracking.data_association import match_3d_2d_detections, match_multicam, CamDetectionIndices
from utils import utils_viz, io
from utils.utils_geometry import clip_bbox_to_four_corners
from utils.transform.transformation import Transformation


""" We have three basic classes: Dataset, Sequence, Frame """


class MOTFrame(ABC):
    """
    base class for frame, used for frame process and instance fuse

    Args:
        sequence (MOTSequence): sequence object
        name (str): frame name

    Examples:
        >>> super().__init__(sequence, name) # recommended to inherit this class
    """
    def __init__(self, sequence: mot_sequence.MOTSequence, name: str):
        self.sequence = sequence
        self.name = name

        # Detections 3D
        self._bboxes_3d: List[Bbox3d] = []
        # Detections 3D
        self._dets_2d_multicam: Dict[str, List[Detection2D]] = {}

        self.instance_fusion_bbox_dir = os.path.join(sequence.instance_fusion_bbox_dir, self.name)

        self.det_score_thresholds = None
        self.seg_score_thresholds = None

        self.fused_instances: List[FusedInstance] = []

        self._raw_pcd = None
        self._points_rect = None
        self.data = None

    @property
    def transformation(self) -> Transformation:
        return self.sequence.transformation

    ##########################################################
    # Detection fusion

    def fuse_instances_and_save(self, params: Mapping, run_info: Dict, *, load: bool = True, save: bool = True):
        """
        fuse 2d and 3d detection result and save

        Args:
            params (dict): algorithm parameters
            run_info (dict): run time scaler
            load (bool): whether load result
            save (bool): whether save result

        Returns:
            run time info
        """
        self.det_score_thresholds = params["det_scores"]
        self.seg_score_thresholds = params["seg_scores"]

        if load:
            dir_to_load = Path(self.get_fused_instances_dir(params))
            if Path.is_dir(dir_to_load):
                self.fused_instances.extend([FusedInstance.load(filename)
                                             for filename in sorted(dir_to_load.iterdir())])
                return

        start_matching = time.time()
        matched_ids, unmatched_3d_ids, unmatched_2d_ids = self.match_3d_2d_dets_with_iou(
            self.bboxes_3d, self.dets_2d_multicam, params["fusion_iou_threshold"])
        matching_t = time.time() - start_matching

        start_creating_instances = time.time()
        self.create_instances_from_matches(matched_ids, unmatched_3d_ids, unmatched_2d_ids)
        creating_instances_t = time.time() - start_creating_instances

        if save:
            self.save_fused_instances_if_new(params)

        # Add FusedInstances stats - how many with both 3D and 2D and only 3D or 2D
        total_unmatched_dets_2d = sum(len(v) for v in unmatched_2d_ids.values())
        if not save:
            run_info["instances_both"] += len(matched_ids)
            run_info["instances_3d"] += len(unmatched_3d_ids)
            run_info["instances_2d"] += total_unmatched_dets_2d
            # To help debug
            # print(f"matched {len(matched_ids)} " +
            #       f"unmatched_det {len(unmatched_3d_ids)} " +
            #       f"unmatched_seg {total_unmatched_dets_2d}")
            run_info["total_time_matching"] += matching_t
            run_info["total_time_creating"] += creating_instances_t
            run_info["total_time_fusion"] += matching_t + creating_instances_t
        return run_info

    def match_3d_2d_dets_with_iou(self, dets_3d: List[Bbox3d], dets_2d_multicam: Mapping[str, List[Detection2D]],
                                  fusion_iou_threshold: float) -> \
                                  Tuple[Dict[int, CamDetectionIndices], Set[int], Dict[str, List[int]]]:
        """ 
        Matches 3D and 2D detections using 2D IoU as the metric. You can follow the code from here to trace the logic

        Args: 
            dets_3d (List): 3d detection result
            dets_2d_multicam (List): 2d (stero) camera detection result
            fusion_iou_threshold (Float): fusion threshold 

        Returns:
            _, unmatched 3d result, unmatched 3d result
        """
        unmatched_dets_3d_ids: Set[int] = set(range(len(dets_3d)))  # default
        if dets_3d and dets_2d_multicam:  # If anything is detected in 3D
            matched_indices: Dict[int, List[CamDetectionIndices]] = defaultdict(list)
            unmatched_dets_2d_ids: Dict[str, List[int]] = defaultdict(list)
            for cam, dets_2d in dets_2d_multicam.items():
                matched_in_cam: Dict[int, CamDetectionIndices] = {}
                unmatched_dets_2d_ids_for_cam: List[int] = []
                if dets_2d:
                    matched_in_cam, _, unmatched_dets_2d_ids_for_cam = \
                        match_3d_2d_detections(dets_3d, cam, dets_2d, fusion_iou_threshold,
                                               self.sequence.classes_to_track)

                for dets_3d_i, cam_det_2d_i in matched_in_cam.items():
                    matched_indices[dets_3d_i].append((cam, cam_det_2d_i))
                unmatched_dets_2d_ids[cam].extend(unmatched_dets_2d_ids_for_cam)

            # remove matched track indices from the unmatched indices set
            unmatched_dets_3d_ids -= matched_indices.keys()

            assert unmatched_dets_3d_ids.union(matched_indices.keys()) == set(range(len(dets_3d)))
            for list_matches in matched_indices.values():
                assert not len(list_matches) > len(self.sequence.cameras)
                for (cam, det_i) in list_matches:
                    assert det_i not in unmatched_dets_2d_ids[cam]

            matched_final_indices = match_multicam(matched_indices, dets_3d)
            assert unmatched_dets_3d_ids.isdisjoint(matched_final_indices.keys())
            return matched_final_indices, unmatched_dets_3d_ids, unmatched_dets_2d_ids
        else:  # nothing is matched
            unmatched_dets_2d_ids = {cam: list(range(len(dets_2d)))
                                     for cam, dets_2d in dets_2d_multicam.items()}
            return {}, unmatched_dets_3d_ids, unmatched_dets_2d_ids

    def create_instances_from_matches(self, matched_indices: Mapping[int, CamDetectionIndices],
                                      unmatched_det_ids: Iterable[int],
                                      unmatched_seg_ids: Mapping[str, Iterable[int]]):
        """ 
        Creates FusedInstance objects given ids of detections that were fused/not fused 
        
        Args:
            matched_indices (CamDetectionIndices): matched indices
            unmatched_det_ids (int): unmatched detection index
            unmatched_seg_ids (int): unmatched segmentation index

        Returns:
            None
        """
        for det_id, (cam, det_2d_id) in matched_indices.items():  # construct fully fused instances
            bbox_3d = self.bboxes_3d[det_id]
            detection_2d = self.dets_2d_multicam[cam][det_2d_id]
            assert bbox_3d.seg_class_id == detection_2d.seg_class_id, \
                f'3D class is {SEG_TO_TRACK_CLASS[bbox_3d.seg_class_id]}, 2D is {SEG_TO_TRACK_CLASS[detection_2d.seg_class_id]}'
            current_object = FusedInstance(len(self.fused_instances),
                                           detection_2d=detection_2d,
                                           bbox_3d=bbox_3d)
            current_object.source = Source.DET_AND_SEG
            current_object.mask_id = det_2d_id
            self.fused_instances.append(current_object)

        for det_3d_id in unmatched_det_ids:
            bbox_3d = self.bboxes_3d[det_3d_id]
            current_object = FusedInstance(len(self.fused_instances),
                                           class_id=bbox_3d.seg_class_id, bbox_3d=bbox_3d)
            current_object.source = Source.DET
            self.fused_instances.append(current_object)

        for cam, det_2d_list in unmatched_seg_ids.items():
            for det_2d_id in det_2d_list:
                current_object = FusedInstance(len(self.fused_instances),
                                               detection_2d=self.dets_2d_multicam[cam][det_2d_id])
                current_object.source = Source.SEG
                current_object.mask_id = det_2d_id
                self.fused_instances.append(current_object)

    def fuse(self, params: Dict, run_info: Dict = defaultdict(int), test_mode: bool=False):
        """
        fuse stage API

        Args:
            params (dict): algorithm parameters
            run_info (dict): run time scaler

        Returns:
            (List): self.fused_instances, params, self.data, run_info, ego_transform, angle_around_y
        """
        run_info=self.fuse_instances_and_save(params, run_info, load=False, save=test_mode)
        # reset KF/tracking coordinates for loaded FusedInstance objects
        for fused_object in self.fused_instances:
            if fused_object.bbox3d is not None:
                fused_object.bbox3d.reset_kf_coordinates()

        ego_transform, angle_around_y = None, None
        if params["compensate_ego"]:
            ego_transform, angle_around_y = self.transform_instances_to_world_frame()
        # if test_mode:
        #     with open("test/fuse_result",'wb') as r:
        #         pickle.dump(self.fused_instances,r)
        return [self.fused_instances, params, self.data, run_info, ego_transform, angle_around_y]

    ##########################################################
    # IO for fused instances

    def save_fused_instances(self, fused_instances_dir):
        """
        save single fuse instances

        Args:
            fused_instances_dir (str): directory to save fused instance

        Returns:
            None
        """
        for instance in self.fused_instances:
            instance.save(fused_instances_dir)

    def save_fused_instances_if_new(self, params: Mapping):
        """
        save fuse instances API

        Args:
            params (dict): algorithm parameters

        Returns:
            None
        """
        fused_instances_dir = self.get_fused_instances_dir(params)
        if io.makedirs_if_new(fused_instances_dir):
            self.save_fused_instances(fused_instances_dir)

    def get_fused_instances_dir(self, params):
        """
        return save directory

        Args:
            params (dict): algorithm parameters

        Returns:
            directory name (str)
        """
        return self.instance_fusion_bbox_dir % (*self.det_score_thresholds, *self.seg_score_thresholds, *params["fusion_iou_threshold"])

    ##########################################################
    # Lazy loading

    @property
    def raw_pcd(self):  # {data:x, extrinsic:x }
        """ 
        return point cloud float64 numpy array 
        
        Args:
            None

        Returns:
            numpy array. 
            shape=(n, 3), reflectance [0, 0.99] 
        """
        if self._raw_pcd is None:
            try:
                self._raw_pcd = self.load_raw_pcd()
            except FileNotFoundError:
                print(f'No point cloud for frame {self.name}')
        return self._raw_pcd

    @property
    def points_rect(self):  # {data:x, extrinsic:x }
        """
        return rectangle point

        Args:
            None

        Returns:
            self._points_rect
        """
        if self._points_rect is None and self.raw_pcd is not None:
            self._points_rect = self.transformation.rect_from_lidar(self.raw_pcd, self.data)
        return self._points_rect

    def points(self, world: bool) -> np.ndarray:
        """
        return points in world Coordinate or rect

        Args:
            world (bool): whether in world coordinate

        Returns:
            points (numpy array)
        """
        return self.points_world if world else self.points_rect

    def load_segmentations_2d_if_needed(self):
        """
        load 2d segmentations if no multicam detection

        Args:
            None

        Returns:
            None
        """
        if not self._dets_2d_multicam:
            dets_2d_multicam = self.sequence.get_segmentations_for_frame(self.name)
            # dets_2d_multicam = self.bbox_2d_annotation_projections()
            self._dets_2d_multicam = {cam: [det for det in dets_2d_multicam[cam]
                                            if det.score > self.seg_score_thresholds[det.seg_class_id - 1]]
                                      for cam in self.sequence.cameras}

    def load_detections_3d_if_needed(self):
        """
        load 3d segmentations if there is 3d detection

        Args:
            None

        Returns:
            None
        """
        if not self._bboxes_3d:
            bboxes_3d = self.sequence.get_bboxes_for_frame(self.name,self.param)
            # bboxes_3d = self.bbox_3d_annotations(world=True)
            self._bboxes_3d = [det for det in bboxes_3d
                               if det.confidence > self.det_score_thresholds[det.seg_class_id - 1]]
            # project 3D bboxes to all cameras
            for bbox_3d in self._bboxes_3d:
                bbox_3d.clear_2d()
                for cam in self.sequence.cameras:
                    bbox_projected = self.transformation.img_from_tracking(bbox_3d.corners_3d, cam, self.data)
                    box_coords = clip_bbox_to_four_corners(
                        bbox_projected, self.sequence.img_shape_per_cam[cam])
                    bbox_3d._bbox_2d_in_cam[cam] = Bbox2d(*box_coords) if box_coords is not None else None

    @property
    def dets_2d_multicam(self) -> Dict[str, List[Detection2D]]:
        self.load_segmentations_2d_if_needed()
        return self._dets_2d_multicam

    @property
    def bboxes_3d(self) -> List[Bbox3d]:
        self.load_detections_3d_if_needed()
        return self._bboxes_3d

    def detections_3d(self, world: bool) -> List[Bbox3d]:
        return self.bboxes_3d_world if world else self.bboxes_3d_ego

    ##########################################################
    # Helpers

    def draw_bounding_boxes(self, det_score_thresholds, seg_score_thresholds, fusion_iou_threshold,
                            cam: str, draw_3d: bool, draw_2d: bool, draw_3d_projection: bool):
        """
        draw bounding boxes for detection

        Args:
            det_score_thresholds (float): detection thresholds
            seg_score_thresholds (float): segmentaion thresholds
            fusion_iou_threshold (float): fusion thresholds
            cam (str): camera name
            draw_3d (bool): whether draw 3d
            draw_2d (bool): whether draw 2d
            draw_3d_projection (bool): whether draw 3d projection

        Returns:
            None
        """
        _params = {"fusion_iou_threshold": fusion_iou_threshold}
        dir_to_save = os.path.join(self.get_fused_instances_dir(_params), 'bboxes')
        io.makedirs_if_new(dir_to_save)

        if self.raw_pcd is None:
            return

        self.det_score_thresholds = det_score_thresholds
        self.seg_score_thresholds = seg_score_thresholds
        img = self.get_image_original_uint8(self.sequence.camera_default)

        if draw_3d:
            if self.bboxes_3d:
                for bbox_3d in self.bboxes_3d:
                    bbox_3d_projected = bbox_3d.bbox_2d_in_cam(cam)
                    utils_viz.draw_bbox(img, bbox_3d_projected, (0, 0, 255), 1)  # Blue

        if draw_2d:
            bboxes_2d_in_cam = self.dets_2d_multicam.get(cam, [])
            for detection_2d in bboxes_2d_in_cam:
                utils_viz.draw_bbox(img, detection_2d.bbox, (255, 0, 0), 1)  # Red

        if draw_3d_projection:
            print(f"frame {self.name}: img_shape_per_cam {self.sequence.img_shape_per_cam}")
            img_default_shape_real = self.sequence.img_shape_per_cam[self.sequence.camera_default]
            if self.bboxes_3d:
                for bbox_3d in self.bboxes_3d:
                    bbox_projected = self.transformation.img_from_tracking(bbox_3d.corners_3d,
                                                                           self.sequence.camera_default, self.data)
                    rect_coords = clip_bbox_to_four_corners(bbox_projected, img_default_shape_real)
                    utils_viz.draw_bbox(img, rect_coords, (0, 0, 0), 2)  # Black

        utils_viz.save_image(img, os.path.join(dir_to_save, self.name + '.png'), convert_to_uint8=False)

    ##########################################################
    # Required methods and fields

    @abstractmethod
    def get_image_original(self, cam: str):
        pass

    @abstractmethod
    def get_image_original_uint8(self, cam: str):
        pass

    @abstractmethod
    def load_raw_pcd(self) -> np.ndarray:
        " Nx3 points"

    @abstractmethod
    def transform_instances_to_world_frame(self) -> Tuple[np.ndarray, float]:
        pass

    @property
    @abstractmethod
    def bboxes_3d_world(self) -> List[Bbox3d]:
        pass

    @property
    @abstractmethod
    def bboxes_3d_ego(self) -> List[Bbox3d]:
        pass

    @property
    @abstractmethod
    def points_world(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def center_world_point(self) -> np.ndarray:
        pass
