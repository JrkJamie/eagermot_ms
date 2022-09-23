import os
import ujson as json
from typing import Any, Dict, Iterable
from utils.fused_instance import FusedInstance
from utils.transform.nuscenes import convert_kitti_bbox_coordinates_to_nu
from typing import List
from enum import Enum


def id_from_name(name: str) -> int:
    return NuScenesClasses[name].value


def name_from_id(class_id: int) -> str:
    return NuScenesClasses(class_id).name


class NuScenesClasses(Enum):
    car = 1
    pedestrian = 2
    bicycle = 3
    bus = 4
    motorcycle = 5
    trailer = 6
    truck = 7


ALL_NUSCENES_CLASS_NAMES: List[str] = [m.name for m in NuScenesClasses]
ALL_NUSCENES_CLASS_IDS: List[int] = [m.value for m in NuScenesClasses]


def build_results_dict(instance: FusedInstance, frame_token: str) -> Dict[str, Any]:
    """
    build nuscenes tracking result in dict

    Args:
        instance (FusedInstance): result instance
        frame_token (str): frame identifier

    Returns:
        track_dict
    """
    assert instance.report_mot
    bbox3d_coords = instance.coordinates_3d  # [h, w, l, x, y, z, theta]
    assert bbox3d_coords is not None
    center, wlh, rotation = convert_kitti_bbox_coordinates_to_nu(bbox3d_coords)
    track_dict: Dict[str, Any] = {"sample_token": frame_token}
    track_dict["translation"] = center.tolist()
    track_dict["size"] = wlh.tolist()
    track_dict["rotation"] = rotation.elements.tolist()
    velocity = instance.bbox3d.velocity
    track_dict["velocity"] = list(velocity) if velocity is not None else [1.0, 1.0]
    track_dict["tracking_id"] = str(instance.track_id)
    track_dict["tracking_name"] = name_from_id(instance.class_id)
    track_dict["tracking_score"] = instance.bbox3d.confidence
    track_dict["yaw"] = bbox3d_coords[6]
    return track_dict


def add_results_to_submit(submission: Dict[str, Dict[str, Any]], frame_token: str,
                          predicted_instances: Iterable[FusedInstance]) -> None:
    """
    add results to submission
    """
    assert frame_token not in submission["results"], submission["results"][frame_token]
    submission["results"][frame_token] = []

    for instance in predicted_instances:
        if instance.report_mot:
            submission["results"][frame_token].append(build_results_dict(instance, frame_token))

    if len(submission["results"][frame_token]) == 0:
        print(f"Nothing tracked for {frame_token}")


def save_to_json_file(submission: Dict[str, Dict[str, Any]],
                      folder_name: str, version: str) -> None:
    """
    low api to save result in json
    """
    print(f"Frames tracked: {len(submission['results'].keys())}")
    results_file = os.path.join(folder_name, (version + "_tracking.json"))
    with open(results_file, 'w') as f:
        json.dump(submission, f, indent=4)
