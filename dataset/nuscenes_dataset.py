from __future__ import annotations
import os
from typing import Optional, List, Dict, Set, Any, Iterable, Sequence, IO, Callable
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
import numpy as np
from collections import defaultdict
import matplotlib.image as mpimg
import imageio
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.eval.common.utils import quaternion_yaw
from utils.utils_geometry import clip_bbox_to_four_corners
from dataset.base_class.mot_dataset import MOTDataset
from dataset.base_class.mot_sequence import MOTSequence
from dataset.base_class.mot_frame import MOTFrame
import dataset.base_class.reporting as reporting
from utils.inputs.bbox import Bbox3d, Bbox2d
from utils.transform.nuscenes import TransformationNuScenes, ROTATION_NEGATIVE_X_FULL
import utils.inputs.loading as loading
from utils.inputs.detection_2d import Detection2D
from utils.fused_instance import FusedInstance


class MOTDatasetNuScenes(MOTDataset):
    """
    MOTDataset for nuscenes, used to load nuscenes dataset and save tracking result

    Args:
        work_dir (str): work directory
        det_source (str): detection directory
        seg_source (str): segmentation directory
        version (str): nuscenes version to run on. e.g. "v1.0-mini", "v1.0-trainval", "v1.0-test"
        data_dir (str): data directory
        param (dict): algorithm parameters
        args (argparse): algorithm settings
    """
    ALL_SPLITS = {"train", "val", "test", "train_detect", "train_track",
                  "mini_train", "mini_val"}

    def __init__(self, work_dir: str, det_source: str, seg_source: str, version: str, data_dir: str, param, args):
        super().__init__(work_dir, det_source, seg_source)
        self.param = param
        self.args = args
        print(f"Parsing NuScenes {version} ...")
        self.nusc = NuScenes(version=version, dataroot=data_dir, verbose=True)
        self.splits: Set[str] = set(s for s in self.ALL_SPLITS if s.split("_")[0] in version)
        self.sequences_by_name: Dict[str, Any] = {
            scene["name"]: scene for scene in self.nusc.scene
        }
        self.splits_to_scene_names: Dict[str, List[str]] = create_splits_scenes()
        print("Done parsing")

        self.version = version
        self.reset()

    def reset(self) -> None:
        """
        reset the NuScenes class settings
        """
        self.submission: Dict[str, Dict[str, Any]] = {"meta": {"use_camera": True,
                                                               "use_lidar": True,
                                                               "use_radar": False,
                                                               "use_map": False,
                                                               "use_external": False},
                                                      "results": {}}
        self.detections_3d: Dict[str, List[Bbox3d]] = {}

    def sequence_names(self, split: str) -> List[str]:
        self.assert_split_exists(split)
        return self.splits_to_scene_names[split]

    def get_sequence(self, split: str, sequence_name: str, save: bool) -> MOTSequenceNuScenes:
        """
        get nuscenes sequences according to name
        """
        self.assert_sequence_in_split_exists(split, sequence_name)
        split_dir = os.path.join(self.work_dir, split)
        return MOTSequenceNuScenes(self.det_source, self.seg_source, split_dir, split,
                                   self.nusc, self.sequences_by_name[sequence_name],
                                   self.submission, self.detections_3d, save_result=save, param=self.param,
                                   args=self.args)

    def save_all_mot_results(self, folder_name: str) -> None:
        """
        api to save nuscenes tracking result in one json file. This should be called after the entire tracking completes
        """
        reporting.save_to_json_file(self.submission, folder_name, self.version)


class MOTSequenceNuScenes(MOTSequence):
    """
    sequence class for nuscenes, used to load nuscenes detection

    Args:
        det_source (str): detection directory
        seg_source (str): segmentation directory
        split_dir (str): split directory
        nusc (NuScenes): nuscenes dataset class
        scene (str): scene name
        dataset_submission (dict): dataset setting
        dataset_detections_3d (dict): dataset detections
        save_result (bool): whether save result
        param (dict): algorithm parameters
        args (argparse): algorithm settings
    """

    def __init__(self, det_source: str, seg_source: str, split_dir: str, split: str,
                 nusc: NuScenes, scene, dataset_submission: Dict[str, Dict[str, Any]],
                 dataset_detections_3d: Dict[str, List[Bbox3d]], save_result, param, args):
        self.param = param
        self.nusc = nusc
        self.scene = scene
        self.args = args
        self.frame_tokens = self._parse_frame_tokens()
        self.dataset_submission = dataset_submission
        self.dataset_detections_3d = dataset_detections_3d
        super().__init__(det_source, seg_source, split_dir,
                         self.scene["name"], self.frame_tokens, save_result=save_result)
        self.data_dir = os.path.join(args.root_dir + args.data_dir + args.dataset, split)
        self._transformation: Optional[TransformationNuScenes] = None
        self.mot.transformation = self.transformation
        fusion_name = 'det_%s_%s_%s_%s_%s_%s_%s_seg_%s_%s_%s_%s_%s_%s_%s_iou_%s_%s_%s_%s_%s_%s_%s'
        self.instance_fusion_bbox_dir = os.path.join(
            self.work_split_input_dir, 'instance_fusion_bbox', fusion_name, self.name)
        self.first_frame = self.nusc.get("sample", self.frame_tokens[0])
        self.token = self.scene["token"]
        self.center_world_point: Optional[np.ndarray] = None

    @property
    def transformation(self) -> TransformationNuScenes:
        if self._transformation is None:
            self._transformation = TransformationNuScenes(self.nusc, self.scene)
        return self._transformation

    def _parse_frame_tokens(self) -> List[str]:
        """
        parse frame tokens one by one
        """
        frame_tokens: List[str] = []
        frame_token = self.scene['first_sample_token']  # first frame token
        while frame_token:  # should break when loading the last frame, which has None for "next"
            frame_nu = self.nusc.get("sample", frame_token)
            frame_tokens.append(frame_token)
            assert frame_nu["scene_token"] == self.scene["token"]
            # update token to the next frame
            frame_token = frame_nu["next"]

        expected_num_frames = self.scene["nbr_samples"]
        assert (len(frame_tokens) ==
                expected_num_frames), f"Expected {expected_num_frames} frames but parsed {len(frame_tokens)}"
        return frame_tokens

    def get_frame(self, frame_token: str) -> MOTFrameNuScenes:
        """
        get frame class

        Args:
            frame_token (str): frame identifier

        Returns:
            MOTFrameNuScenes
        """
        frame = MOTFrameNuScenes(self, frame_token, self.nusc, self.param, self.args)
        if not self.img_shape_per_cam:
            for cam in self.cameras:
                frame.get_image_original(cam)
            self.mot.img_shape_per_cam = self.img_shape_per_cam
        return frame

    def load_detections_3d(self) -> Dict[str, List[Bbox3d]]:
        """
        load 3d detections
        """
        if not self.dataset_detections_3d:
            self.dataset_detections_3d.update(
                loading.load_detections_3d(self.det_source, self.name, self.param, self.args))
        return self.dataset_detections_3d

    def load_detections_2d(self) -> Dict[str, Dict[str, List[Detection2D]]]:
        """
        load 2d detections
        """
        frames_cam_tokens_detections = loading.load_detections_2d_nuscenes(self.seg_source, self.token, self.param,
                                                                           self.args)
        frames_cams_detections: Dict[str, Dict[str, List[Detection2D]]
        ] = defaultdict(lambda: defaultdict(list))

        for frame_token, cam_detections in frames_cam_tokens_detections.items():
            for cam_data_token, detections in cam_detections.items():
                cam = self.nusc.get('sample_data', cam_data_token)["channel"]
                for detection in detections:
                    detection.cam = cam
                frames_cams_detections[frame_token][cam] = detections
        return frames_cams_detections

    @property
    def cameras(self) -> List[str]:
        return ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
                "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

    @property
    def camera_default(self) -> str:
        return "CAM_FRONT"

    @property
    def classes_to_track(self) -> List[int]:
        return reporting.ALL_NUSCENES_CLASS_IDS

    def report_mot_results(self, frame_name: str, predicted_instances: Iterable[FusedInstance],
                           mot_3d_file: IO, mot_2d_from_3d_only_file: Optional[IO]) -> None:
        reporting.add_results_to_submit(self.dataset_submission, frame_name, predicted_instances)

    def save_mot_results(self, mot_3d_file: IO, mot_2d_from_3d_file: Optional[IO]) -> None:
        pass

    def load_ego_motion_transforms(self) -> None:
        """ Not needed for NuScenes """

    def save_ego_motion_transforms_if_new(self) -> None:
        """ Not needed for NuScenes """


AB3DMOT = 'ab3dmot'
CENTER_POINT = 'center_point'
EFFICIENT_DET = 'efficient_det'


class MOTFrameNuScenes(MOTFrame):
    """
    frame class for nuscenes, used for coordinate transform

    Args:
        sequence (MOSequence): sequence this frame belongs to
        name (str): name of frame
        nusc (NuScenes): nuscenes dataset class
        param (dict): algorithm parameters
        args (argparse): algorithm settings
    """
    def __init__(self, sequence, name: str, nusc: NuScenes, param, args):
        super().__init__(sequence, name)
        self.nusc = nusc
        self.param = param
        self.args = args
        self.frame = self.nusc.get("sample", name)
        assert self.frame["scene_token"] == self.sequence.scene["token"]
        self.data = self.frame["data"]
        self._points_world: Optional[np.ndarray] = None

    @property
    def transformation(self) -> TransformationNuScenes:
        return self.sequence.transformation

    def get_image_original(self, cam: str):
        return self._read_image(cam, mpimg.imread)

    def get_image_original_uint8(self, cam: str):
        return self._read_image(cam, imageio.imread)

    def _read_image(self, cam: str, read_function: Callable):
        """
        api to read 2d images
        """
        image_path = self.nusc.get_sample_data_path(self.data[cam])
        image = read_function(image_path)
        # need to remember actual image size
        self.sequence.img_shape_per_cam[cam] = image.shape[:2]
        return image

    def load_raw_pcd(self):
        """
        api to load 3d points in pcd
        """
        lidar_data = self.nusc.get('sample_data', self.data["LIDAR_TOP"])
        assert lidar_data["is_key_frame"]
        lidar_filepath = os.path.join(self.nusc.dataroot, lidar_data["filename"])
        nu_pcd: LidarPointCloud = LidarPointCloud.from_file(lidar_filepath)
        return nu_pcd.points[:3].T

    @property
    def points_world(self) -> np.ndarray:
        """
        return data in world coordinate
        """
        if self._points_world is None:
            self._points_world = self.transformation.world_from_lidar(
                self.raw_pcd, self.data)
        if self.sequence.center_world_point is None:
            self.sequence.center_world_point = self._points_world.mean(axis=0)
            self.sequence.center_world_point[2] = 0.0
        return self._points_world

    @property
    def center_world_point(self) -> np.ndarray:
        if self.sequence.center_world_point is None:
            self.points_world
        return self.sequence.center_world_point

    def bbox_3d_annotations(self, world: bool = False) -> List[Bbox3d]:
        """
        return 3d bbox in list
        """
        bboxes = (self.bbox_3d_annotation(token, world) for token in self.frame["anns"])
        return [bbox for bbox in bboxes if bbox is not None]

    def bbox_3d_annotation(self, annotation_token: str, world: bool = False) -> Optional[Bbox3d]:
        """
        get 3d bbox annotation
        """
        bbox_nu = self.nusc.get_box(annotation_token)  # annotations are in world coordinates
        if not world:
            bbox_nu = self.transformation.ego_box_from_world(bbox_nu, self.data)
        bbox_nu.score = 1.0
        bbox_nu.velocity = [1.0, 1.0]

        instance_id = hash(annotation_token)
        name_parts = bbox_nu.name.split(".")
        bbox_class = name_parts[1] if len(name_parts) > 1 else name_parts[0]
        if bbox_class in reporting.ALL_NUSCENES_CLASS_NAMES:
            return Bbox3d.from_nu_box_convert(bbox_nu, instance_id)
        else:
            return None

    def bbox_2d_annotation_projections(self) -> Dict[str, List[Detection2D]]:
        """
        project 3d annotation to 2d
        """
        # use annotation projections
        dets_2d_multicam: Dict[str, List[Detection2D]] = {cam: [] for cam in self.sequence.cameras}
        bboxes_3d = self.bbox_3d_annotations(world=True)
        for bbox_3d in bboxes_3d:
            for cam in self.sequence.cameras:
                bbox_projected = self.transformation.img_from_tracking(bbox_3d.corners_3d, cam, self.data)
                box_coords = clip_bbox_to_four_corners(
                    bbox_projected, self.sequence.img_shape_per_cam[cam])
                if box_coords is not None:
                    dets_2d_multicam[cam].append(Detection2D(
                        Bbox2d(*box_coords), cam, bbox_3d.confidence, bbox_3d.seg_class_id))
        return dets_2d_multicam

    @property
    def bboxes_3d_world(self) -> List[Bbox3d]:
        return self.bboxes_3d

    @property
    def bboxes_3d_ego(self) -> List[Bbox3d]:
        """
        transform 3d bbox according to ego motion
        """
        lidar_data = self.nusc.get('sample_data', self.data["LIDAR_TOP"])
        ego_pose_data = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        transform_matrix = np.ones((4, 4), float)
        rotation_quaternion = Quaternion(ego_pose_data["rotation"])
        transform_matrix[:3, :3] = rotation_quaternion.rotation_matrix
        transform_matrix[:3, 3] = ego_pose_data["translation"]
        angle_around_vertical = -1 * quaternion_yaw(rotation_quaternion)

        bboxes = self.bboxes_3d.copy()
        for bbox in bboxes:
            # need to go back to Nu coordinates
            bbox.inverse_transform(ROTATION_NEGATIVE_X_FULL, 0)
            # transform to ego
            bbox.inverse_transform(transform_matrix, angle_around_vertical)
            # back to internal tracking frame, i.e. KITTI's original/ego frame
            bbox.transform(ROTATION_NEGATIVE_X_FULL, 0)
        return bboxes

    def transform_instances_to_world_frame(self):
        return None, None
