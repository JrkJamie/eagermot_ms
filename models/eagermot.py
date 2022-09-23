import time
from blocks.match_3d import Match3d
from blocks.match_2d import Match2d
import numpy as np

class Eagermot:
    """
    class to start algorithm

    Args:
        sequence (MOTSequence): sequence to track
        params (dict): params (dict): algorithm parameters
    """

    def __init__(self, sequence, params, test_mode=False):
        self.sequence = sequence
        self.params = params
        self.test_mode= test_mode
        sequence.mot.set_track_manager_params(self.params)
        # self.block1=block2.match3d()

    def run(self):
        """
        start the algorithm using this function

        Args:
            None

        Returns:
            run_info
        """
        # create result file and run_info in dataset/base/mot_sequence 
        run_info, mot_3d_file, mot_2d_from_3d_file, flag = self.sequence.preprocess(self.params)
        if not flag:
            return run_info
        self.run_info = run_info
        # iterate frame
        for frame_i, frame_name in enumerate(self.sequence.frame_names):
            if frame_i % 100 == 0:
                print(f'Processing frame {frame_name}')
            frame = self.sequence.get_frame(frame_name)

            # as pre-stage, we fuse instance first
            fuse_result = frame.fuse(self.params, self.run_info, self.test_mode)
            self.params = fuse_result[1]
            self.run_info = fuse_result[3]
            start_mot_ego = time.time()

            # start stage1 and stage2
            self.sequence.mot.frame_count += 1
            for track in self.sequence.mot.trackers:
                track.reset_for_new_frame()
            run_info['reset_time'] += time.time() - start_mot_ego
            # iterate classes    
            for class_target in self.sequence.mot.classes_to_track:
                # init track
                track_empty = self.sequence.mot.init_track(frame.fused_instances, class_target)

                # stage1
                stage1 = Match3d(track_empty[0], track_empty[2], self.params)
                result1 = stage1.run()
                if frame.name is "000152" and self.test_mode:
                    np.save("example/eagermot/test/result/result3d.npy",result1)
                # stage2
                stage2 = Match2d(track_empty, result1, self.run_info, fuse_result, self.sequence.mot,
                                 frame.data)
                result2 = stage2.run(self.sequence.mot)
                if frame.name is "000152" and self.test_mode:
                    np.save("example/eagermot/test/result/result2d.npy",result1)
                self.run_info = result2[-1]

                # update track
                self.sequence.mot.update_track(track_empty, result1, result2, class_target)

            # report tracks
            ego_transform = fuse_result[-2]
            angle_around_y = fuse_result[-1]
            predicted_instances = self.sequence.mot.report_tracks(ego_transform, angle_around_y)
            self.sequence.mot.remove_obsolete_tracks()
            self.run_info["total_time_mot"] += time.time() - start_mot_ego
            predicted_instances.sort(key=lambda x: x.distance_to_ego)

            # account time and report
            start_reporting = time.time()
            self.sequence.report_mot_results(frame.name, predicted_instances, mot_3d_file, mot_2d_from_3d_file)
            self.run_info["total_time_reporting"] += time.time() - start_reporting

        # save result
        self.sequence.save_mot_results(mot_3d_file, mot_2d_from_3d_file)
        self.sequence.save_ego_motion_transforms_if_new()
        return self.run_info
