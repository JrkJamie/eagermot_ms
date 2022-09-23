from __future__ import annotations
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from utils.transform.transformation import inverse_rigid_transform
import utils.inputs.bbox as bbox
import utils.inputs.detection_2d as detection_2d
import utils.inputs.detections_2d as detections_2d
from enum import Enum


class Source(Enum):
    DET = 1
    SEG = 2
    DET_AND_SEG = 3
    IGNORE = -1


class FusedInstance(bbox.ProjectsToCam):
    """
    class to represent fused instance

    Args:
        class_id (int): class id
        detection_2d (Detection2D): 2d detection
        bbox_3d (Bbox3d): 3d detection
    """

    def __init__(self, instance_id,
                 class_id: Optional[int] = None,
                 detection_2d: Optional[detection_2d.Detection2D] = None,
                 bbox_3d: Optional[bbox.Bbox3d] = None):
        self.instance_id: int = instance_id
        self.class_id: Optional[int] = class_id
        self.points_rect = None
        self.colors = None
        self.source = Source.IGNORE

        # Info from 3D detections
        self.bbox3d = bbox_3d

        # Info from 2D detections
        self.mask_id = detections_2d.NO_LABEL
        self.detection_2d = detection_2d
        self.class_id = detection_2d.seg_class_id if detection_2d is not None else class_id

        # Info derived from the best of the two sources
        self.bbox_2d_source: Optional[Source] = None
        self.bbox_2d_conf: Optional[float] = None

        # Track-related info
        self.track_id = None
        self.can_have_mask_from_points = None

        # Fileds only used for reporting final results
        self.coordinates_3d = None
        self.report_mot = False
        self.report_mots = False

        self.projected_bbox_3d = None

    def set_with_instance_from_mask(self, instance_from_mask):
        """
        set instance from mask

        Args:
            instance_from_mask (FusedInstance): instance

        Returns:
            None
        """
        self.detection_2d = instance_from_mask.detection_2d
        self.mask_id = instance_from_mask.mask_id
        self.source = Source.SEG
        self.bbox_2d_conf = instance_from_mask.bbox_2d_conf

    def reset_seg(self, keep_matching_info=False):
        """
        reset segmentation

        Args:
            keep_matching_info (bool): whether keep match info

        Returns:
            None
        """
        self.mask_id = detections_2d.NO_LABEL
        if self.detection_2d:
            self.detection_2d.mask = None
            if not keep_matching_info:
                self.detection_2d.score = None
                self.detection_2d.reid = None
                self.detection_2d.bbox = None

    def bbox_2d_best(self, cam) -> Optional[bbox.Bbox2d]:
        """
        return best bbox 2d, else None

        Args:
            cam (str): camera name

        Returns:
            Bbox2d
        """
        if self.bbox3d is not None and self.bbox3d.bbox_2d_in_cam(cam) is not None:
            self.bbox_2d_source = Source.DET
            self.bbox_2d_conf = float(self.bbox3d.confidence)
            return self.bbox3d.bbox_2d_in_cam(cam)

        bbox_from_2d: Optional[bbox.Bbox2d] = self.bbox_2d_from_2d_in_cam(cam)
        if bbox_from_2d is not None:
            self.bbox_2d_source = Source.SEG
            self.bbox_2d_conf = self.score
            return bbox_from_2d

        self.bbox_2d_source = Source.IGNORE
        self.bbox_2d_conf = 0
        return None

    def bbox_2d_from_2d_in_cam(self, cam: str) -> Optional[bbox.Bbox2d]:
        return self.detection_2d.bbox_2d_in_cam(cam) if self.detection_2d else None

    def bbox_2d_in_cam(self, cam: str) -> Optional[bbox.Bbox2d]:
        return self.bbox_2d_from_2d_in_cam(cam)
        # raise NotImplementedError  # do not use this method, the name is ambiguous

    @property
    def distance_to_ego(self):
        if not self.bbox3d is not None:
            offset = self.score if self.score else 0
            return 1e10 - offset
        return np.linalg.norm(self.bbox3d.centroid_original[(0, 2),])  # only x-z plane - ignore elevation

    def transform(self, transformation, angle_around_y=None):
        assert transformation is not None, 'Requested None transformation'
        if angle_around_y is not None and self.bbox3d is not None:
            self.bbox3d.transform(transformation, angle_around_y)

    def inverse_transform(self, transformation, angle_around_y):
        assert transformation is not None, 'Requested None reverse transformation'
        self.transform(inverse_rigid_transform(transformation), -angle_around_y)

    def save(self, path_to_frame_folder):
        with open(Path(path_to_frame_folder) / str(self.instance_id), 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path_to_file):
        with open(Path(path_to_file), 'rb') as handle:
            return pickle.load(handle)

    @property
    def bbox_2d(self) -> Optional[bbox.Bbox2d]:
        return self.detection_2d.bbox if self.detection_2d else None

    @property
    def score(self) -> Optional[float]:
        return self.detection_2d.score if self.detection_2d else None

    @property
    def mask(self):
        return self.detection_2d.mask if self.detection_2d else None

    @property
    def reid(self):
        return self.detection_2d.reid if self.detection_2d else None
