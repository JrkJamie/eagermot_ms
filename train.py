import run_tracking, argparse, yaml
from pathlib import Path


def train_algothm(args, param):
    train_param = ["fusion_iou_threshold", "thresholds_per_class", "leftover_matching_thres", "max_ages"]

    # result saved in utils/test_result
    # best param below
    # [0.03,0.03],[-3.8,-0.6], 0.2, [2,2] 

    ind = 3
    param['kitti'][train_param[ind]][1] = 2
    param['kitti'][train_param[ind]][0] = 2
    for i in range(4):
        # param['kitti'][train_param[ind]][1]+=1
        # param['kitti'][train_param[ind]][1]+=0.01
        run_info, val = run_tracking.run_on_kitti(args, param)
        param1 = param
        out = open(str(Path(__file__).resolve().parents[1]) + "/utils/test_result/" + train_param[ind] + "_" + str(
            param['kitti'][train_param[ind]]) + ".yaml", 'w+', encoding='utf-8"')
        # del_list=['fusion_mode','is_angular','compensate_ego','iou_3d_threshold']
        # for item in del_list:
        #     del param1['kitti'][str(item)]
        param1['result'] = dict()
        add_list = ['first_matched_percentage', 'second_matched_percentage', 'second_matched_dets2d_second_percentage',
                    'final_unmatched_percentage']
        for i, key in enumerate(add_list):
            param1['result'][key] = val[i]
        yaml.dump(param1, out)
        out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="/data0/HR_dataset/JIANG/EagerMOT", type=str, help="root dir")
    parser.add_argument("--split", default="/training", type=str)
    parser.add_argument("--work_dir", default="/storage/workspace/kitti", type=str)
    parser.add_argument("--data_dir", default="/storage/datasets/kitti")
    parser.add_argument("--result_dir", default="/result", type=str)
    parser.add_argument("--param_dir", default="/data0/HR_dataset/JIANG/eagermot/utils/configs/param.yaml", type=str)
    parser.add_argument("--save_result", default=True, type=bool)
    args = parser.parse_args()
    file = open(args.param_dir, 'r', encoding="utf-8")
    param = yaml.load(file, Loader=yaml.FullLoader)
    file.close()

    # train params
    args.save_result = False
    train_algothm(args, param)
