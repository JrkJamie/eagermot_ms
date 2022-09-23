from typing import Iterable, List, Dict, Optional, Sequence, Any
import numpy as np

from utils.tracking.tracks import Track
from utils.transform.transformation import Transformation
from utils.utils_geometry import project_bbox_3d_to_2d
from utils.inputs.bbox import Bbox2d


class TrackManager(object):
    """
    class managing tracks

    Args:
        cameras (str): camera names
        classes_to_track (int): class to track
    """
    def __init__(self, cameras: Sequence[str], classes_to_track: Iterable[int]):
        self.trackers: List[Track] = []
        self.frame_count = 0
        self.cameras = cameras
        self.classes_to_track = classes_to_track

        Track.count = 0
        self.track_ids_map: Dict[int, int] = {}
        self.track_id_latest = 1  # evaluations expect positive track ids

        # will be set by the calling MOTSequence
        self.transformation: Optional[Transformation] = None
        # maps cameras to their image plane shapes whose len need to be 2: tuple, array, etc.
        self.img_shape_per_cam: Optional[Dict[str, Any]] = None

    def set_track_manager_params(self, params):
        '''
        set parameters from params

        Args:
            params (dict): algorithm parameters

        Returns:
            None
        '''
        self.is_angular = params.get('is_angular', False)
        # How many frames a track gets to match with something before it is killed (2?)
        self.max_ages = params.get('max_ages')
        self.max_age_2d = params.get("max_age_2d")
        # How many matches a track needs to get in its lifetime to be considered confirmed
        self.min_hits = params.get('min_hits')

        self.second_matching_method = params.get('second_matching_method', 'iou')
        self.leftover_thres = params.get('leftover_matching_thres')

    def init_track(self,fused_instances,class_target):
        '''
        initialize tracks

        Args:
            fused_instances (instance): fused instance in pre-stage
            class_target (int): the class index
        
        Returns:
            tracks
        '''
        det_instances_3d = [instance for instance in fused_instances
                                if instance.bbox3d is not None and instance.class_id == class_target]
        det_instances_from_mask = [instance for instance in fused_instances
                                    if not instance.bbox3d is not None and instance.class_id == class_target]

        tracks_with_3d_models = []
        track_indices_with_motion_models = []
        track_indices_without_motion_models = []
        for track_i, track in enumerate(self.trackers):
            if track.class_id != class_target:
                continue

            if track.has_motion_model:
                tracks_with_3d_models.append(track)
                track_indices_with_motion_models.append(track_i)
            else:
                track_indices_without_motion_models.append(track_i)
        return [det_instances_3d,det_instances_from_mask,tracks_with_3d_models,track_indices_with_motion_models,track_indices_without_motion_models]

    def update_track(self,track_empty,result1,result2,class_target):
        '''
        update tracks based on result
        
        Args:
            track_empty (list): initial track
            result1 (list): first stage result
            result2 (list): second stage result
            class_target (int): class to track

        Returns:
            None
        '''
        # update tracks that were matched with fully fused instances
        det_instances_3d=track_empty[0]
        matched_instances_to_tracks_first=result1[0]
        leftover_track_indices=result2[0];matched_tracks_to_cam_instances_second=result2[1];leftover_tracks=result2[2];leftover_det_instance_multicam=result2[3]
        unmatched_det_indices_final=result2[4];leftover_det_instances_no_2d=result2[5]
        for track_i in matched_instances_to_tracks_first[:, 1]:
            assert track_i not in leftover_track_indices
            track = self.trackers[track_i]
            assert track.class_id == class_target
            matched_det_id = matched_instances_to_tracks_first[np.where(
                matched_instances_to_tracks_first[:, 1] == track_i)[0], 0]
            matched_instance = det_instances_3d[matched_det_id[0]]
            track.update_with_match(matched_instance)

        # update tracks that were watched with instances based on 2D IoU (may or may not have 3D boxes)
        for track_i_secondary, (cam, instance_i) in matched_tracks_to_cam_instances_second.items():
            track = leftover_tracks[track_i_secondary]
            assert track.class_id == class_target
            track.update_with_match(leftover_det_instance_multicam[cam][instance_i])

        # create and initialise new tracks for all unmatched detections
        for cam, indices in unmatched_det_indices_final.items():
            for instance_i in indices:
                instance = leftover_det_instance_multicam[cam][instance_i]
                if instance.bbox3d is not None:
                    self.trackers.append(Track(instance, self.is_angular))
        self.trackers.extend([Track(instance, self.is_angular)
                                for instance in leftover_det_instances_no_2d if instance.bbox3d is not None])

                                
    def report_tracks(self, ego_transform, angle_around_y):
        '''
        gather and report final tracks

        Args:
            ego_transform (numpy): 4x4 transformation matrix to convert from current to world coordinates
            angle_around_y (float): defaults to None
        
        Returns:
            track result
        '''
        # from [x,y,z,theta,l,w,h] to [h, w, l, x, y, z, theta]
        reorder_back = [6, 5, 4, 0, 1, 2, 3]
        instances_tracked = []
        for track in reversed(self.trackers):
            if self.is_recent(track.class_id, track.time_since_update):
                track_id = self.unique_track_id(track.id)
                instance = track.current_instance(
                    ego_transform, angle_around_y, self.min_hits[track.class_id - 1])
                instance.track_id = track_id

                instance.report_mot = (self.is_confirmed_track(track.class_id, track.hits, track.age_total)
                                       and track.time_since_update == 0)

                if track.has_motion_model:  # report MOT
                    bbox_3d = track.current_bbox_3d(ego_transform, angle_around_y)  # current 3D bbox
                    instance.coordinates_3d = bbox_3d.kf_coordinates[reorder_back]
                    instance.bbox3d.obs_angle = track.obs_angle

                    if len(self.cameras) < 2:  # KITTI
                        bbox_2d = project_bbox_3d_to_2d(
                            bbox_3d, self.transformation, self.img_shape_per_cam, self.cameras[0], None)
                        instance.projected_bbox_3d = Bbox2d(*bbox_2d) if bbox_2d is not None else None

                    max_age_2d_for_class = self.max_age_2d[track.class_id - 1]
                    if track.time_since_2d_update < max_age_2d_for_class:
                        instance.bbox3d.confidence = track.confidence
                    else:
                        frames_since_allowed_no_2d_update = track.time_since_2d_update + 1 - max_age_2d_for_class
                        instance.bbox3d.confidence = track.confidence / \
                            (2.0 * frames_since_allowed_no_2d_update)
                else:
                    instance.report_mot = False

                instances_tracked.append(instance)
        return instances_tracked

    def remove_obsolete_tracks(self):
        '''
        remove old tracks

        Args:
            None
        
        Returns:
            None
        '''
        track_i = len(self.trackers) - 1
        for track in reversed(self.trackers):
            if track.time_since_update >= self.max_ages[track.class_id - 1]:
                del self.trackers[track_i]
            track_i -= 1

    def is_confirmed_track(self, class_id, hits, age_total):
        '''
        whether track is confirmed

        Args:
            class_id (int): class id
            hits (int): number hits
            age_total (int): frame tracked
        
        Returns:
            result (bool)
        '''
        required_hits = self.min_hits[class_id - 1]
        if self.frame_count < required_hits:
            return hits >= age_total
        else:
            return hits >= required_hits

    def is_recent(self, class_id, time_since_update):
        '''
        whether track is recent

        Args:
            class_id (int): class id
            time_since_update (int): time after update
        
        Returns:
            result (bool)
        '''
        return time_since_update < self.max_ages[class_id - 1]

    def unique_track_id(self, original_track_id):  
        '''
        this was a necessary workaround for MOTS, which has a limit on submitted track IDs
        If the submitted track.id was larger than some threshold, evaluation did not work correctly
        This might have been patched since then, we alerted the MOTS team, but this might be safer still

        Args:
            original_track_id (int): original_track_id

        Returns:
            processed index
        '''
        if original_track_id not in self.track_ids_map:
            self.track_ids_map[original_track_id] = self.track_id_latest
            self.track_id_latest += 1
        return self.track_ids_map[original_track_id]
