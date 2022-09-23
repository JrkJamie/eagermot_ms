import pickle
import numpy as np
import argparse, yaml, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]) + "/models")
sys.path.append(str(Path(__file__).resolve().parents[1]))
from dataset.kitti_dataset import MOTDatasetKITTI
from eagermot import Eagermot
from blocks.match_3d import Match3d
from blocks.match_2d import Match2d
from utils.fused_instance import FusedInstance


def load_model(param, args):
    mot_dataset = MOTDatasetKITTI(work_dir=args.root_dir + args.work_dir + args.dataset,
                                  det_source=param['kitti']['AB3DMOT'],
                                  seg_source=param['kitti']['TRACKING_BEST'],
                                  data_dir=args.root_dir + args.data_dir + args.dataset, param=param['kitti'],
                                  args=args)
    model=Eagermot(mot_dataset.get_sequence("training", "0000", True),
                    param['kitti'], test_mode=True)
    model.sequence.preprocess(model.params)
    frame = model.sequence.get_frame("000152")
    return model, frame


def test_fuse(model, frame, ignore=False):
    """
    统一测试kitti 第一个序列倒数第二帧
    """
    [now_fuse, _, _, _, _, _] = frame.fuse(model.params, dict(), True)
    # with open("example/eagermot/test/result/fuse",'wb') as r:
    #     pickle.dump(origin_fuse, r)
    if not ignore:
        with open("example/eagermot/test/result/fuse",'rb') as f:
            origin_fuse=pickle.load(f)
            f.close()
        
        for i in range(len(origin_fuse)):
            if(origin_fuse[i].bbox3d and origin_fuse[i].detection_2d):
                assert np.allclose(origin_fuse[i].bbox3d.original_coordinates,now_fuse[i].bbox3d.original_coordinates), "fuse 3d bbox error"
                assert np.allclose(origin_fuse[i].detection_2d.bbox,now_fuse[i].detection_2d.bbox), "fuse 2d bbox error"
            assert np.allclose(origin_fuse[i].instance_id,now_fuse[i].instance_id), "fuse id error"
            assert isinstance(now_fuse[i], FusedInstance), "data type error"
        print("################################")
        print("fuse test passed!")
        print("################################")
    return now_fuse


def test_match3d(result1, model, ignore=False):
    track_empty=model.sequence.mot.init_track(result1, 1)
    stage1 = Match3d(track_empty[0], track_empty[2], model.params)
    result2=stage1.run()
    if not ignore:
        gt_result2=np.load("example/eagermot/test/result/match3d.npy",allow_pickle=True)
        assert np.allclose(gt_result2[0],result2[0]), "matched error"
        assert np.allclose(gt_result2[1],result2[1]), "undet error"
        assert np.allclose(gt_result2[2],result2[2]), "unmotion error"
        print("################################")
        print("match3d test passed!")
        print("################################")
    return result2, track_empty


def test_match2d(result2, track_empty, fuse_result):
    stage2 = Match2d(track_empty, result2, dict(), fuse_result, model.sequence.mot, frame.data, True)
    result2 = stage2.run(model.sequence.mot)
    gt_result2=np.load("example/eagermot/test/result/match2d.npy", allow_pickle=True)
    for ind, tracks in enumerate(result2):
        for ind2, track in enumerate(tracks):
            if track.bbox3d and track.detection_2d:
                assert np.allclose(track.bbox3d.original_coordinates,gt_result2[ind][ind2].bbox3d.original_coordinates), "match2d bbox3d error"
                assert np.allclose(track.detection_2d.bbox,gt_result2[ind][ind2].detection_2d.bbox), "match2d bbox2d error"
            assert np.allclose(track.instance_id,gt_result2[ind][ind2].instance_id), "match2d id error"
    print("################################")
    print("match2d test passed!")
    print("################################")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="this is a module test script", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset", default="/kitti", type=str, help="test dataset /kitti or /nuscenes")
    parser.add_argument("--split", default="/training", type=str,
                        help="the split to test on. such as /training, /testing")
    parser.add_argument("--result_dir", default="/result", type=str, help="directory to store result under work_dir")
    parser.add_argument("--root_dir", default="/data0/HR_dataset/JIANG/EagerMOT", type=str,
                        help="root directory for the workspace")
    parser.add_argument("--work_dir", default="/storage/workspace", type=str, help="work directory under root_dir")
    parser.add_argument("--data_dir", default="/storage/datasets", help="dataset directory under root_dir")
    parser.add_argument("--param_dir", default="example/eagermot/utils/configs/param.yaml",
                        type=str, help="directory to store param file")
    parser.add_argument("--ignore_result", default=True, type=bool,
                        help="whether to ignore previous results, if true, the new results will cover previous ones")
    parser.add_argument("--test_mode",default=False, help="whether to test module, sqecify the target sequence 0000")
    parser.add_argument("--module_name",default="all", help="fuse, match3d, match2d, all")
    args = parser.parse_args()
    file = open(args.param_dir, 'r', encoding="utf-8")
    param = yaml.load(file, Loader=yaml.FullLoader)
    file.close()
    model, frame=load_model(param, args)
    if args.module_name is "fuse":
        test_fuse(model, frame)
    elif args.module_name is "match3d":
        result1=test_fuse(model, frame, True)
        test_match3d(result1, model)
    elif args.module_name is "match2d":
        result1=test_fuse(model, frame, True)
        result2, track_empty=test_match3d(result1, model, True)
        test_match2d(result2, track_empty, result1)
    elif args.module_name is "all":
        result1=test_fuse(model, frame)
        result2, track_empty=test_match3d(result1, model)
        test_match2d(result2, track_empty, result1)
