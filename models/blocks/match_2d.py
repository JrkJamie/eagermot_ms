from typing import List, Dict, Set, Iterable, Any, Mapping, Sequence
from collections import defaultdict
from utils.fused_instance import FusedInstance
from typing import Tuple
from utils.tracking.tracks import Track
from utils.utils_geometry import box_2d_area
from utils.inputs.bbox import ProjectsToCam
from utils.tracking.data_association import associate_boxes_2d

class Match2d:
    """
    class for second stage 2d match

    Args:
        track_empty (list): scalers about instances
        result1 (list): first stage result
        run_info (dict): run time result
        fuse_result (list): fuse result
        mot (TrackManager): track manager of this sequence
        frame_data (list): frame data
    """

    CamDetectionIndices = Tuple[str, int]

    def __init__(self, track_empty, result1, run_info, fuse_result, mot, frame_data, test_mode=False) -> None:
        self.fuse_result = fuse_result
        self.frame_data = frame_data
        self.run_info = run_info
        self.test_mode=test_mode
        # map indices of motion tracks back to original indices among all tracks
        matched_instances_to_tracks_first = result1[0]
        unmatched_det_indices_first = result1[1]
        unmatched_motion_track_indices_first = result1[2]
        det_instances_3d = track_empty[0]
        det_instances_from_mask = track_empty[1]
        track_indices_with_motion_models = track_empty[3]
        track_indices_without_motion_models = track_empty[-1]
        for i in range(len(matched_instances_to_tracks_first)):
            matched_instances_to_tracks_first[i, 1] = \
                track_indices_with_motion_models[matched_instances_to_tracks_first[i, 1]]

        # Gather tracks that have no motion model
        self.leftover_track_indices = [track_indices_with_motion_models[i]
                                       for i in unmatched_motion_track_indices_first]
        self.leftover_track_indices.extend(track_indices_without_motion_models)
        self.leftover_tracks = [mot.trackers[track_i] for track_i in self.leftover_track_indices]
        if not test_mode:
            self.run_info["matched_tracks_first_total"] += len(matched_instances_to_tracks_first)
            self.run_info["unmatched_tracks_first_total"] += len(self.leftover_tracks)

        assert len(unmatched_det_indices_first) == len(set(unmatched_det_indices_first))
        assert len(unmatched_motion_track_indices_first) == len(set(unmatched_motion_track_indices_first))
        assert len(self.leftover_track_indices) == len(set(self.leftover_track_indices))

        # Gather all unmatched detected instances (no 3D box + failed 1st stage)
        self.leftover_det_instance_multicam: Dict[str, List[FusedInstance]] = defaultdict(list)
        self.leftover_det_instances_no_2d: List[FusedInstance] = []
        self.total_leftover_det_instances = len(det_instances_from_mask)
        for instance in det_instances_from_mask:
            assert instance.detection_2d
            self.leftover_det_instance_multicam[instance.detection_2d.cam].append(instance)
        for det_i in unmatched_det_indices_first:
            instance = det_instances_3d[det_i]
            # #431 do not use instances with 3D in the 2nd stage
            if instance.detection_2d and instance.bbox3d is None:
                self.leftover_det_instance_multicam[instance.detection_2d.cam].append(instance)
                self.total_leftover_det_instances += 1
            else:
                self.leftover_det_instances_no_2d.append(instance)

    def run(self, mot):
        '''
        function to start the second stage

        Args:
            mot (TrackManager): track manager of this sequence

        Returns:
            a list containing results
        '''
        matched_indices: Dict[int, List[self.CamDetectionIndices]] = defaultdict(list)
        unmatched_track_indices_final: Set[int] = set(range(len(self.leftover_tracks)))
        unmatched_det_indices_final: Dict[str, Set[int]] = {
            cam: set(range(len(det_instances))) for cam, det_instances in self.leftover_det_instance_multicam.items()
        }
        ego_transform = self.fuse_result[-2];
        angle_around_y = self.fuse_result[-1]
        if mot.leftover_thres is not None and mot.leftover_thres < 1.0:
            # second_start_time = time.time()
            for cam, instances_list in self.leftover_det_instance_multicam.items():
                assert mot.second_matching_method == "iou"
                assert all(instance.bbox3d is None for instance in instances_list)  # 431
                (matched_instances_to_tracks_second_cam, unmatched_det_indices_cam,
                 unmatched_track_indices_cam) = \
                    self.associate_instances_to_tracks_2d_iou(
                        instances_list, self.leftover_tracks, mot.leftover_thres,
                        ego_transform, angle_around_y, mot.transformation, mot.img_shape_per_cam, cam, self.frame_data)

                for instance_i, track_i in matched_instances_to_tracks_second_cam:
                    matched_indices[track_i].append((cam, instance_i))
                    unmatched_det_indices_final[cam].discard(instance_i)
            # print(f"2nd stage: {time.time() - second_start_time:.2f}")
            # print()
        # remove matched track indices from the unmatched indices set
        unmatched_track_indices_final -= matched_indices.keys()
        # print(f"matched_indices:\n{matched_indices}")
        # print(f"matched_indices.keys():\n{matched_indices.keys()}")
        # print(f"unmatched_track_indices_final:\n{unmatched_track_indices_final}")
        # print(f"unmatched_det_indices_final:\n{unmatched_det_indices_final}")
        assert unmatched_track_indices_final.union(
            matched_indices.keys()) == set(range(len(self.leftover_tracks)))

        total_matched_leftover_instances = 0
        for list_matches in matched_indices.values():
            assert not len(list_matches) > len(mot.cameras)
            total_matched_leftover_instances += len(list_matches)
            for (cam, det_i) in list_matches:
                assert det_i not in unmatched_det_indices_final[cam]

        total_unmatched_leftover_instances = 0
        for unmatched_det_indices in unmatched_det_indices_final.values():
            total_unmatched_leftover_instances += len(unmatched_det_indices)
        assert total_matched_leftover_instances + total_unmatched_leftover_instances == self.total_leftover_det_instances

        matched_tracks_to_cam_instances_second = self.match_multicam(matched_indices, self.leftover_tracks)
        if not self.test_mode:
            self.run_info["matched_tracks_second_total"] += len(matched_tracks_to_cam_instances_second)
            self.run_info["unmatched_tracks_second_total"] += len(unmatched_track_indices_final)
            self.run_info["unmatched_dets2d_second_total"] += total_unmatched_leftover_instances

        return [self.leftover_track_indices, matched_tracks_to_cam_instances_second, self.leftover_tracks,
                self.leftover_det_instance_multicam, unmatched_det_indices_final, self.leftover_det_instances_no_2d,
                self.run_info]

    def associate_instances_to_tracks_2d_iou(self, instances_leftover: Iterable[FusedInstance],
                                         tracks_leftover: Iterable[Track],
                                         iou_threshold: float,
                                         ego_transform, angle_around_y,
                                         transformation, img_shape_per_cam: Mapping[str, Any],
                                         cam: str, frame_data: Mapping[str, Any]):
        """
        high API to assign detected_objects to tracked objects according to 2d iou

        Args:
            instances_leftover (FusedInstance): instances leftover
            tracks_leftover (Track): tracks leftover
            iou_threshold (dict): iou threshold
            ego_transform (numpy): ego transform
            angle_around_y (float): angle around y
            transformation (object): transformation
            img_shape_per_cam (str): img shape per cam
            cam (str): camera name
            frame_data (str): frame raw data

        Returns:
            3 lists of matches, unmatched_detections and unmatched_trackers
        """
        detected_bboxes_2d = [instance.bbox_2d_best(cam) for instance in instances_leftover]
        tracked_bboxes_2d = [track.predicted_bbox_2d_in_cam(ego_transform, angle_around_y,
                                                            transformation, img_shape_per_cam,
                                                            cam, frame_data) for track in tracks_leftover]
        return associate_boxes_2d(detected_bboxes_2d, tracked_bboxes_2d, iou_threshold)

    def match_multicam(self, candidate_matches: Mapping[int, Sequence[CamDetectionIndices]],
                   instances_3d: Sequence[ProjectsToCam]) -> Dict[int, CamDetectionIndices]:
        """ 
        Matches each 3D instance (3D detection / track) with a single 2D detection given 
        a list of possible detections to match to. Decides which candidate detection to assign
        based on the area of the 2D projection of the 3D instance in that camera.
        The intuition is that the cam where the 3D object is most prevalent 
        will most likely have its 2D projection recognized correctly

        Args:
            candidate_matches (CamDetectionIndices): maps entities to a *sequence* of possible 2D detections
            instances_3d (ProjectsToCam): entities that need unique matches
            
        Returns:
            dict mapping original entities to a *single* 2D detection
        """
        matched_indices: Dict[int, self.CamDetectionIndices] = {}
        for instance_i, cam_det_2d_indices in candidate_matches.items():
            assert cam_det_2d_indices  # has to be at least 1 candidate match
            if len(cam_det_2d_indices) == 1:
                matched_indices[instance_i] = cam_det_2d_indices[0]
            else:
                # if matches were made in multiple cameras,
                # select the camera with the largest projection of the 3D detection
                instance_3d = instances_3d[instance_i]
                largest_area = 0.0

                for cam, det_2d_i in cam_det_2d_indices:
                    area = box_2d_area(instance_3d.bbox_2d_in_cam(cam))
                    assert area > 0, f"All of the candidate 2D projections have to be valid {instance_3d.bbox_2d_in_cam(cam)}"
                    if area > largest_area:
                        largest_area = area
                        chosen_cam_det = (cam, det_2d_i)
                assert largest_area > 0, "3D instance has to have at least one valid 2D projection"
                matched_indices[instance_i] = chosen_cam_det
                # assuming that matches were correct in multiple cameras
                # simply discard duplicate 2D detections and don't treat them as unmatched
                # another option is to allow multiple 2D detections for each 3D instance and ignore this function
                #
                # if matches were incorrect, then these "duplicates" should be added
                # to the rest of unmatched detections, but we will assume matches are correct
        return matched_indices
