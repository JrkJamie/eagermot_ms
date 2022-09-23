from __future__ import annotations
from ast import arg
import glob
import os
from enum import Enum
from typing import Optional, Dict, List, Set, Iterable, IO, Tuple, Callable
import matplotlib.image as mpimg
import imageio
import numpy as np
import utils.inputs.detections_2d as detections_2d
import dataset.base_class.mot_dataset as mot_dataset
import dataset.base_class.mot_sequence as mot_sequence
import dataset.base_class.mot_frame as mot_frame
import utils.io as io
import utils.inputs.loading as loading
import utils.inputs.utils as utils
from utils.inputs.detections_2d import CAR_CLASS, PED_CLASS
from utils.inputs.bbox import Bbox2d, Bbox3d
from utils.inputs.detection_2d import Detection2D
from utils.fused_instance import FusedInstance
from utils.transform.kitti import TransformationKitti
from utils.transform.transformation import to_homogeneous
from utils.utils_geometry import angles_from_rotation_matrix


class KITTIClasses(Enum):
    car = 1
    pedestrian = 2


ALL_KITTI_CLASS_NAMES: List[str] = [m.name for m in KITTIClasses]
ALL_KITTI_CLASS_IDS: List[int] = [m.value for m in KITTIClasses]


def name_from_id(class_id: int) -> str:
    return KITTIClasses(class_id).name


def id_from_name(name: str) -> int:
    return KITTIClasses[name].value


class MOTFrameKITTI(mot_frame.MOTFrame):
    """
    kitti frame class, used for frame process

    Args:
        sequence (MOSequence): sequence this frame belongs to
        name (str): name of frame
    """
    def __init__(self, sequence, name, param, args):
        super().__init__(sequence, name)
        self.param = param
        self.args = args
        self.image_path = os.path.join(sequence.image_path, f'{name}.png')
        self.pcd_path = os.path.join(sequence.pcd_path, f'{name}.bin')
        self._ego_transform: Optional[np.ndarray] = None
        self._angle_around_vertical: Optional[float] = None

    def get_image_original(self, cam: str = "image_02"):
        """
        get image using mpimg

        Args:
            cam (str): camera selection. default for image_02

        Returns:
            image
        """
        return self._read_image(cam, mpimg.imread)

    def get_image_original_uint8(self, cam: str = "image_02"):
        """
        get image using imageio

        Args:
            cam (str): camera selection. default for image_02

        Returns:
            image
        """
        return self._read_image(cam, imageio.imread)

    def _read_image(self, cam: str, read_function: Callable):
        """
        get image using specified function

        Args:
            cam (str): camera selection
            read_function (function): module to read image

        Returns:
            image

        Examples:
            >>> _read_image(cam, imageio.imread)
        """
        image = read_function(self.image_path % cam)
        # need to remember actual image size
        self.sequence.img_shape_per_cam[cam] = image.shape[:2]
        return image

    def load_raw_pcd(self):
        """
        load pcd file from self.pcd_path

        Args:
            None

        Returns:
            None
        """
        pcd = np.fromfile(self.pcd_path, dtype=np.float32)
        return pcd.reshape((-1, 4))[:, :3]

    @property
    def points_world(self):
        """
        return points in world coordinate

        Args:
            None

        Returns:
            points (numpy)
        """
        points = (self.ego_transform @ to_homogeneous(self.points_rect).T).T
        return points[:, :-1]

    @property
    def center_world_point(self) -> np.ndarray:
        return np.zeros((1, 3), dtype=float)  # 3D points are already centered

    ##########################################################
    # Ego motion

    @property
    def ego_transform(self):
        """
        apply left-side to go from ego to world

        Args:
            None

        Returns:
            ego_transform
        """
        if self._ego_transform is None:
            self._ego_transform = self.sequence.ego_transform_for_frame(self.name)
        return self._ego_transform

    @property
    def angle_around_vertical(self):
        if self._angle_around_vertical is None:
            self._angle_around_vertical = angles_from_rotation_matrix(self.ego_transform[:3, :3])[1]
        return self._angle_around_vertical

    @property
    def images_for_vo(self):
        images = [mpimg.imread(self.image_path % 'image_02'), mpimg.imread(self.image_path % 'image_03')]
        images = [np.mean(x, -1) for x in images]
        return images

    def transform_instances_to_world_frame(self) -> Tuple[np.ndarray, float]:
        """
        transform_instances_to_world_frame

        Args:
            None

        Returns:
            ego_transform, angle_around_vertical
        """
        for fused_object in self.fused_instances:
            fused_object.transform(self.ego_transform, self.angle_around_vertical)
        return self.ego_transform, self.angle_around_vertical

    @property
    def bboxes_3d_ego(self) -> List[Bbox3d]:
        return self.bboxes_3d

    @property
    def bboxes_3d_world(self):
        bboxes = self.bboxes_3d.copy()
        for bbox in bboxes:
            bbox.transform(self.ego_transform, self.angle_around_vertical)
        return bboxes

    def bbox_3d_annotations(self, world: bool = False) -> List[Bbox3d]:  # List[Box]
        """
        get bounding box annotation

        Args:
            world (bool): whether in world coordinate

        Returns:
            bbox_3d_annotations
        """
        assert not world  # kitti only provides annotations in the ego frame
        return self.sequence.bbox_3d_annotations[self.name]

    def bbox_2d_annotation_projections(self) -> Dict[str, List[Detection2D]]:
        """
        project bbox

        Args:
            None

        Returns:
            Detection2D
        """
        cam = "image_02"
        return {cam: [Detection2D(bbox_3d.bbox_2d_in_cam(cam), cam, bbox_3d.confidence,
                                  bbox_3d.seg_class_id) for bbox_3d in self.bbox_3d_annotations(False)]}


class MOTSequenceKITTI(mot_sequence.MOTSequence):
    """
    sequence class for kitti, used to load kitti detection and save results

    Args:
        det_source (str): detection directory
        seg_source (str): segmentation directory
        split_dir (str): split directory
        split (str): split name
        name (str): sequence name
        frame_names (str): frame names under sequence
        save_result (bool): whether save result
    """

    def __init__(self, det_source: str, seg_source: str, split_dir: str, split: str,
                 name: str, frame_names: List[str], save_result: bool, param, args):
        super().__init__(det_source, seg_source, split_dir, name, frame_names, save_result)
        self.data_dir = os.path.join(args.root_dir + args.data_dir + args.dataset, split)
        self.args = args
        self.image_path = os.path.join(self.data_dir, "%s", self.name)
        self.pcd_path = os.path.join(self.data_dir, "velodyne", self.name)
        ego_motion_folder = os.path.join(split_dir, "ego_motion")
        io.makedirs_if_new(ego_motion_folder)
        self.param = param
        self.ego_motion_filepath = os.path.join(ego_motion_folder, self.name + ".npy")
        self.save = save_result
        self._transformation: Optional[TransformationKitti] = None
        self.mot.transformation = self.transformation
        self.transform_accumulated = None
        self.ego_motion_transforms = np.ones(shape=(len(self.frame_names), 4, 4), dtype=np.float)
        self.has_full_ego_motion_transforms_loaded = False

        fusion_name = 'det_%s_%s_seg_%s_%s_iou_%s_%s'
        self.instance_fusion_bbox_dir = os.path.join(
            self.work_split_input_dir, 'instance_fusion_bbox', fusion_name, self.name)

        self._bbox_3d_annotations: Dict[str, List[Bbox3d]] = {}

    @property
    def transformation(self) -> TransformationKitti:
        if self._transformation is None:
            self._transformation = TransformationKitti(self.data_dir, self.name)
        return self._transformation

    def get_frame(self, frame_name: str):
        """
        get frame object

        Args:
            frame_name (str): name of frame

        Returns:
            frame (MOTFrameKITTI)
        """
        frame = MOTFrameKITTI(self, frame_name, self.param, self.args)
        if not self.img_shape_per_cam:  # get the true cam image shape by loading a frame for the first time
            for cam in self.cameras:
                frame.get_image_original(cam)
            self.mot.img_shape_per_cam = self.img_shape_per_cam
        return frame

    def load_detections_3d(self) -> Dict[str, List[Bbox3d]]:
        """
        load 3d detections

        Args:
            None

        Returns:
            bbox 3d (Dict[str, List[Bbox3d]])
        """
        bboxes_3d_all = loading.load_detections_3d(self.det_source, self.name, self.param, self.args)
        return {str(frame_i).zfill(6): bboxes_3d
                for frame_i, bboxes_3d in enumerate(bboxes_3d_all)}

    def load_detections_2d(self) -> Dict[str, Dict[str, List[Detection2D]]]:
        """
        load 2d detections

        Args:
            None

        Returns:
            Detection2D (Dict[str, Dict[str, List[Detection2D]]])
        """
        # if self.seg_source == self.param['MMDETECTION_CASCADE_NUIMAGES']:
        #     return loading.load_detections_2d_kitti_new(self.seg_source, self.name)

        # Load and construct 2D Detections for this sequence, sorted by score ascending
        bboxes_all, scores_all, reids_all, classes_all, masks_all = loading.load_detections_2d_kitti(
            self.seg_source, self.name, self.param, self.args)

        return {str(frame_i).zfill(6): self.construct_detections_2d(bboxes, scores, classes, masks, reids)
                for frame_i, (bboxes, scores, classes, masks, reids)
                in enumerate(zip(bboxes_all, scores_all, classes_all, masks_all, reids_all))}

    def construct_detections_2d(self,
                                bboxes,
                                scores,
                                classes,
                                masks,
                                reids) -> Dict[str, List[Detection2D]]:
        """
        zip and sort 2d detections

        Args:
            bboxes (object): bounding box
            scores (object): score
            classes (int): class id
            masks (object): mask
            reids (object): reids

        Returns:
            dets (Dict[str, List[Detection2D]])
        """
        dets = [Detection2D(Bbox2d(*box), self.cameras[0], score, seg_class_id, mask=mask, reid=reid) for
                box, score, seg_class_id, mask, reid
                in zip(bboxes, scores, classes, masks, reids)]
        dets.sort(key=lambda x: x.score)  # sort detections by ascending score
        return {self.cameras[0]: dets}

    @property
    def bbox_3d_annotations(self) -> Dict[str, List[Bbox3d]]:
        if not self._bbox_3d_annotations:
            self._bbox_3d_annotations = loading.load_annotations_kitti(self.name)
        return self._bbox_3d_annotations

    ##########################################################
    # Ego motion

    def ego_transform_for_frame(self, frame_name: str) -> np.ndarray:
        """
        frame_name: for which frame to get the transformation, only works consecutively

        Args:
            frame_name (str): frame name

        Returns:
            ego_motion_transforms (numpy)
            full transform 4x4 from the current frame/3D pose, in the first frame's coordinate system
        """
        assert self.has_full_ego_motion_transforms_loaded
        frame_int = int(frame_name)
        return self.ego_motion_transforms[frame_int]

    def save_ego_motion_transforms_if_new(self) -> None:
        """
        save ego motion transforms if it's new

        Args:
            None

        Returns:
            None
        """
        if not self.has_full_ego_motion_transforms_loaded:
            with open(self.ego_motion_filepath, 'wb') as np_file:
                np.save(np_file, self.ego_motion_transforms)

    def load_ego_motion_transforms(self) -> None:
        """
        load ego transform to self.ego_motion_transforms

        Args:
            None

        Returns:
            None
        """
        # print(self.ego_motion_filepath)
        assert os.path.isfile(self.ego_motion_filepath), "Missing ego motion files"
        with open(self.ego_motion_filepath, 'rb') as np_file:
            self.ego_motion_transforms = np.load(np_file)
        self.has_full_ego_motion_transforms_loaded = True

    def report_mot_results(self, frame_name: str, predicted_instances: Iterable[FusedInstance],
                           mot_3d_file: IO,
                           mot_2d_from_3d_only_file: Optional[IO]) -> None:
        """
        save mot result

        Args:
            frame_name (str): frame name
            predicted_instances (FusedInstance): mot result
            mot_3d_file (IO): 3d file
            mot_2d_from_3d_only_file (IO): 2d file

        Returns:
            None
        """
        write_to_mot_file(frame_name, predicted_instances, mot_3d_file, mot_2d_from_3d_only_file)

    def save_mot_results(self, mot_3d_file: IO, mot_2d_from_3d_file: Optional[IO]) -> None:
        io.close_files((mot_3d_file, mot_2d_from_3d_file))

    ##########################################################
    # Cameras

    @property
    def camera_params(self):
        return MOTDatasetKITTI.CAMERA_PARAMS

    @property
    def cameras(self) -> List[str]:
        return ["image_02"]  # "image_03"

    @property
    def camera_default(self) -> str:
        return "image_02"

    @property
    def classes_to_track(self) -> List[int]:
        return [CAR_CLASS, PED_CLASS]


class MOTDatasetKITTI(mot_dataset.MOTDataset):
    """
    dataset class for kitti, used to load kitti sequences

    Args:
        work_dir (str): work directory
        det_source (str): detection directory
        seg_source (str): segmentation directory
        data_dir (str): data directory
        param (dict): algorithm parameters
        args (dict): algorithm settings
    """
    FOCAL = 721.537700
    CU = 609.559300
    CV = 172.854000
    BASELINE = 0.532719
    CAMERA_PARAMS = [FOCAL, CU, CV, BASELINE]

    def __init__(self, work_dir, det_source: str, seg_source: str, data_dir: str, param, args):
        super().__init__(work_dir, det_source, seg_source)
        self.splits: Set[str] = {"training", "testing"}
        self.split_sequence_frame_names_map: Dict[str, Dict[str, List[str]]] = {sp: {} for sp in self.splits}
        self.param = param
        self.args = args
        for split in self.splits:
            seq_dir = os.path.join(data_dir, split, 'image_02')
            if not os.path.isdir(seq_dir):
                raise NotADirectoryError(seq_dir)

            # Parse sequences
            for sequence in sorted(os.listdir(seq_dir)):
                img_dir = os.path.join(seq_dir, sequence)
                if os.path.isdir(img_dir):
                    images = glob.glob(os.path.join(img_dir, '*.png'))
                    self.split_sequence_frame_names_map[split][sequence] = [os.path.splitext(os.path.basename(image))[0]
                                                                            for image in sorted(images)]

    def sequence_names(self, split: str) -> List[str]:
        """
        get sequence names

        Args:
            split (str): split name

        Returns:
            sequence names (List[str])
        """
        self.assert_split_exists(split)
        return list(self.split_sequence_frame_names_map[split].keys())

    def get_sequence(self, split: str, sequence_name: str, save_result: bool) -> MOTSequenceKITTI:
        """
        get sequence object

        Args:
            split (str): split name
            sequence_name (str): sequence name
            save_result (bool): whether save result

        Returns:
            sequence object (MOTSequenceKITTI)
        """
        self.assert_sequence_in_split_exists(split, sequence_name)
        split_dir = os.path.join(self.work_dir, split)
        return MOTSequenceKITTI(self.det_source, self.seg_source, split_dir, split, sequence_name,
                                self.split_sequence_frame_names_map[split][sequence_name], save_result,
                                param=self.param, args=self.args)

    def save_all_mot_results(self, folder_name: str) -> None:
        """ KITTI saves results per-sequence, so this method does not apply here """
        pass


def write_to_mot_file(frame_name: str,
                      predicted_instances: Iterable[FusedInstance],
                      mot_3d_file: IO,
                      mot_2d_from_3d_only_file: Optional[IO]) -> None:
    """
    write mot result in format

    Args:
        frame_name (str): frame name
        predicted_instances (FusedInstance): mot result
        mot_3d_file (IO): 3d file
        mot_2d_from_3d_only_file (IO): 2d file

    Returns:
        None
    """
    mot_3d_results_str, mot_2d_results_str = "", ""
    tracking_3d_format = "%d %d %s 0 0 %f -1 -1 -1 -1 %f %f %f %f %f %f %f %f\n"
    tracking_2d_format = "%d %d %s 0 0 -10 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 %f\n"

    for instance in predicted_instances:
        if not instance.report_mot:
            continue

        assert instance.class_id is not None
        track_type = detections_2d.SEG_TO_TRACK_CLASS[instance.class_id]

        bbox3d_coords = instance.coordinates_3d
        if bbox3d_coords is not None:
            bbox3d = instance.bbox3d

            res_3d = (tracking_3d_format % (int(frame_name), instance.track_id, track_type, bbox3d.obs_angle,
                                            bbox3d_coords[0], bbox3d_coords[1], bbox3d_coords[2],
                                            bbox3d_coords[3], bbox3d_coords[4], bbox3d_coords[5],
                                            bbox3d_coords[6], bbox3d.confidence))
            mot_3d_results_str += res_3d

        if mot_2d_from_3d_only_file is not None:
            bbox2d = instance.projected_bbox_3d
            if bbox2d is not None:
                res_2d = (tracking_2d_format % (int(frame_name), instance.track_id, track_type,
                                                bbox2d[0], bbox2d[1], bbox2d[2], bbox2d[3],
                                                instance.bbox3d.confidence))
                mot_2d_results_str += res_2d

    mot_3d_file.write(mot_3d_results_str)
    if mot_2d_from_3d_only_file is not None:
        mot_2d_from_3d_only_file.write(mot_2d_results_str)
