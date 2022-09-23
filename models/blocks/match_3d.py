from typing import Tuple, List
import numpy as np
from scipy.optimize import linear_sum_assignment
from utils.utils_geometry import convert_bbox_coordinates_to_corners, box_2d_area
from utils.tracking.utils_association import (iou_bbox_2d_matrix, iou_bbox_3d_matrix,
                                              distance_2d_matrix, distance_2d_dims_matrix, distance_2d_full_matrix)


class Match3d:
    """
    class for first stage 3d match

    Args:
        det_instances_3d (det): det_instances_3d
        tracks_with_3d_models (track): tracks_with_3d_models
        params (dict): algorithm parameters
    """

    def __init__(self, det_instances_3d, tracks_with_3d_models, params) -> None:
        self.params = params
        self.det_instances_3d = det_instances_3d
        self.tracks_with_3d_models = tracks_with_3d_models

    def run(self):
        """
        start first stage matching

        Args:
            None

        Returns:
            None
        """
        tracks = self.tracks_with_3d_models
        detected_instances = self.det_instances_3d
        params = self.params
        if len(tracks) == 0:  # original association
            return np.empty((0, 2), dtype=int), list(range(len(detected_instances))), []
        if len(detected_instances) == 0:  # nothing detected in the current frame
            return np.empty((0, 2), dtype=int), [], list(range(len(tracks)))
        track_coordinates = [track.predict_motion()[:7] for track in tracks]
        track_classes = [track.class_id for track in tracks]
        if params['first_matching_method'] == 'iou_3d':
            detected_corners = [instance.bbox3d.corners_3d for instance in detected_instances]
            tracks_corners = [convert_bbox_coordinates_to_corners(state) for state in track_coordinates]
            detections_dims = [instance.bbox3d.kf_coordinates[4:7] for instance in detected_instances]
            tracks_dims = [state[4:7] for state in track_coordinates]
            matrix_3d_sim = iou_bbox_3d_matrix(detected_corners, tracks_corners, detections_dims, tracks_dims)
        elif params['first_matching_method'] == "dist_2d":
            detected_centers = [instance.bbox3d.kf_coordinates[:3] for instance in detected_instances]
            track_centers = [state[:3] for state in track_coordinates]
            matrix_3d_sim = distance_2d_matrix(detected_centers, track_centers)
            matrix_3d_sim *= -1
        elif params['first_matching_method'] == "dist_2d_dims":
            detected_coordinates = [instance.bbox3d.kf_coordinates for instance in detected_instances]
            matrix_3d_sim = distance_2d_dims_matrix(detected_coordinates, track_coordinates)
            matrix_3d_sim *= -1
        elif params['first_matching_method'] == "dist_2d_full":
            detected_coordinates = [instance.bbox3d.kf_coordinates for instance in detected_instances]
            matrix_3d_sim = distance_2d_full_matrix(detected_coordinates, track_coordinates)
            matrix_3d_sim *= -1

        matched_indices, unmatched_det_ids, unmatched_track_ids = \
            self._perform_association_from_cost_hu(len(detected_instances), len(tracks), matrix_3d_sim)
        matched_instances_to_tracks_first, unmatched_det_indices_first, unmatched_motion_track_indices_first = \
            self.filter_matches(
                matched_indices, unmatched_det_ids, unmatched_track_ids, matrix_3d_sim,
                params["iou_3d_threshold"],
                track_classes, params["thresholds_per_class"])
        
        return [matched_instances_to_tracks_first, unmatched_det_indices_first, unmatched_motion_track_indices_first]

    def filter_matches(self, matched_indices, unmatched_first_indices, unmatched_second_indices, matrix,
                       threshold='None', classes_second=None, thresholds_per_class=None):
        assert threshold == 'None' or not thresholds_per_class
        matches = []

        if threshold != 'None':
            for m in matched_indices:
                if matrix[m[0], m[1]] < threshold:
                    unmatched_first_indices.append(m[0])
                    unmatched_second_indices.append(m[1])
                else:
                    matches.append(m.reshape(1, 2))
        else:
            for m in matched_indices:
                second_i = m[1]
                if matrix[m[0], second_i] < thresholds_per_class[classes_second[second_i]]:
                    unmatched_first_indices.append(m[0])
                    unmatched_second_indices.append(m[1])
                else:
                    matches.append(m.reshape(1, 2))

        matches = np.vstack(matches) if len(matches) > 0 else np.empty((0, 2), dtype=int)
        return matches, unmatched_first_indices, unmatched_second_indices

    def _perform_association_from_cost_hu(self, first_items_len, second_items_len, cost_matrix
                                          ) -> Tuple[np.ndarray, List[int], List[int]]:
        # Run the Hungarian algorithm for assignment from the cost matrix
        matched_indices = linear_sum_assignment(cost_matrix)
        matched_indices = np.asarray(matched_indices).T

        unmatched_first_items = [i for i in range(first_items_len) if i not in matched_indices[:, 0]]
        unmatched_second_items = [i for i in range(second_items_len) if i not in matched_indices[:, 1]]
        return matched_indices, unmatched_first_items, unmatched_second_items

    def filter_matches(self, matched_indices, unmatched_first_indices, unmatched_second_indices, matrix,
                       threshold='None', classes_second=None, thresholds_per_class=None):
        assert threshold == 'None' or not thresholds_per_class
        matches = []

        if threshold != 'None':
            for m in matched_indices:
                if matrix[m[0], m[1]] < threshold:
                    unmatched_first_indices.append(m[0])
                    unmatched_second_indices.append(m[1])
                else:
                    matches.append(m.reshape(1, 2))
        else:
            for m in matched_indices:
                second_i = m[1]
                if matrix[m[0], second_i] < thresholds_per_class[classes_second[second_i]]:
                    unmatched_first_indices.append(m[0])
                    unmatched_second_indices.append(m[1])
                else:
                    matches.append(m.reshape(1, 2))

        matches = np.vstack(matches) if len(matches) > 0 else np.empty((0, 2), dtype=int)

        return matches, unmatched_first_indices, unmatched_second_indices
