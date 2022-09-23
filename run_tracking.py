import argparse
import sys
import time
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]) + "/models")
sys.path.append(str(Path(__file__).resolve().parents[0]))
from dataset.kitti_dataset import MOTDatasetKITTI
from dataset.nuscenes_dataset import MOTDatasetNuScenes
from eagermot import Eagermot


def report(run_info):
    """
    print algorithm result

    Args:
        run_info: run time result

    Returns:
        run_info, matching percentages
    """
    total_instances = run_info['instances_both'] + run_info['instances_3d'] + run_info['instances_2d']
    if total_instances > 0:
        print(f"Fusion\nTotal instances 3D and 2D: {run_info['instances_both']} " +
              f"-> {100.0 * run_info['instances_both'] / total_instances:.2f}%")
        print(f"Total instances 3D only  : {run_info['instances_3d']} " +
              f"-> {100.0 * run_info['instances_3d'] / total_instances:.2f}%")
        print(f"Total instances 2D only  : {run_info['instances_2d']} " +
              f"-> {100.0 * run_info['instances_2d'] / total_instances:.2f}%")
        print()
    # Matching stats
    print(f"matched_tracks_first_total {run_info['matched_tracks_first_total']}")
    print(f"unmatched_tracks_first_total {run_info['unmatched_tracks_first_total']}")

    print(f"matched_tracks_second_total {run_info['matched_tracks_second_total']}")
    print(f"unmatched_tracks_second_total {run_info['unmatched_tracks_second_total']}")
    print(f"unmatched_dets2d_second_total {run_info['unmatched_dets2d_second_total']}")

    first_matched_percentage = (run_info['matched_tracks_first_total'] /
                                (run_info['unmatched_tracks_first_total'] + run_info['matched_tracks_first_total']))
    print(f"percentage of all tracks matched in 1st stage {100.0 * first_matched_percentage:.2f}%")

    second_matched_percentage = (
            run_info['matched_tracks_second_total'] / run_info['unmatched_tracks_first_total'])
    print(f"percentage of leftover tracks matched in 2nd stage {100.0 * second_matched_percentage:.2f}%")

    second_matched_dets2d_second_percentage = (run_info['matched_tracks_second_total'] / (
            run_info['unmatched_dets2d_second_total'] + run_info['matched_tracks_second_total']))
    print(f"percentage dets 2D matched in 2nd stage {100.0 * second_matched_dets2d_second_percentage:.2f}%")

    final_unmatched_percentage = (run_info['unmatched_tracks_second_total'] / (
            run_info['matched_tracks_first_total'] + run_info['unmatched_tracks_first_total']))
    print(f"percentage tracks unmatched after both stages {100.0 * final_unmatched_percentage:.2f}%")

    print(f"\n3D MOT saved in {run_info['mot_3d_file']}", end="\n\n")
    return run_info, [first_matched_percentage, second_matched_percentage, second_matched_dets2d_second_percentage,
                      final_unmatched_percentage]


def run_on_nuscenes(args, param):
    mot_dataset = MOTDatasetNuScenes(work_dir=args.root_dir + args.work_dir + args.dataset,
                                     det_source=param['nuscenes']['CENTER_POINT'],
                                     seg_source=param['nuscenes']['MMDETECTION_CASCADE_NUIMAGES'],
                                     version=param['nuscenes']['version'],
                                     data_dir=args.root_dir + args.data_dir + args.dataset, param=param['nuscenes'],
                                     args=args)
    start_time = time.time()
    target_sequences = mot_dataset.sequence_names(args.split[1:])

    # specify sequence you want to test on (test only)
    # target_sequences=["scene-0103"]

    # iterate sequence
    for sequence_name in target_sequences:
        print(f'Starting sequence: {sequence_name}')
        model = Eagermot(mot_dataset.get_sequence(args.split[1:], sequence_name, args.ignore_result),
                         param['nuscenes'], test_mode=args.test_mode)
        run_info = model.run()
    print(f'Variant took {(time.time() - start_time) / 60.0:.2f} mins')
    mot_dataset.save_all_mot_results(args.root_dir + args.work_dir + args.dataset + args.split + args.result_dir)
    mot_dataset.reset()

    # print result
    return report(run_info)


def run_on_kitti(args, param):
    mot_dataset = MOTDatasetKITTI(work_dir=args.root_dir + args.work_dir + args.dataset,
                                  det_source=param['kitti']['AB3DMOT'],
                                  seg_source=param['kitti']['TRACKING_BEST'],
                                  data_dir=args.root_dir + args.data_dir + args.dataset, param=param['kitti'],
                                  args=args)
    target_sequences = mot_dataset.sequence_names(args.split[1:])

    # specify sequence you want to test on (test only)
    target_sequences=["0000"]
    start_time = time.time()

    # iterate sequence
    for sequence_name in target_sequences:
        print(f'Starting sequence: {sequence_name}')
        model = Eagermot(mot_dataset.get_sequence(args.split[1:], sequence_name, args.ignore_result),
                         param['kitti'], test_mode=args.test_mode)
        run_info = model.run()
    print(f'Variant took {(time.time() - start_time) / 60.0:.2f} mins')

    # print result
    return report(run_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     "this is a mindspore version of eagermot \n\
 to change dataset between kitti and nuscenes, use argument --dataset\n\
 to change split (training or testing for kitti, train or test for nuscenes), use argument --split\n\
 note that nuscenes require extra modification in utils/configs/param.yaml nuscenes/version\n\
 to change 2d and 3d source, change det_source(3d) and seg_source(2d) in run_on_kitti(nuscenes)\n",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset", default="/kitti", type=str, help="test dataset /kitti or /nuscenes")
    parser.add_argument("--split", default="/training", type=str,
                        help="the split to test on. such as /training, /testing")
    parser.add_argument("--result_dir", default="/result", type=str, help="directory to store result under work_dir")
    parser.add_argument("--root_dir", default="/data0/HR_dataset/JIANG/EagerMOT", type=str,
                        help="root directory for the workspace")
    parser.add_argument("--work_dir", default="/storage/workspace", type=str, help="work directory under root_dir")
    parser.add_argument("--data_dir", default="/storage/datasets", help="dataset directory under root_dir")
    parser.add_argument("--param_dir", default="/data0/HR_dataset/JIANG/ms3d/example/eagermot/utils/configs/param.yaml",
                        type=str, help="directory to store param file")
    parser.add_argument("--ignore_result", default=True, type=bool,
                        help="whether to ignore previous results, if true, the new results will cover previous ones")
    parser.add_argument("--test_mode",default=False, help="whether to test module, sqecify the target sequence 0000")
    args = parser.parse_args()
    file = open(args.param_dir, 'r', encoding="utf-8")
    param = yaml.load(file, Loader=yaml.FullLoader)
    file.close()

    # run algorithm
    if args.dataset == "/kitti":
        run_on_kitti(args, param)
    else:
        run_on_nuscenes(args, param)
