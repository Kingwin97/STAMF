import os.path as osp
from .evaluator import Eval_thread
from .dataloader import EvalDataset


def evaluate(args):

    # parser.add_argument('--Evaluation', default=False, type=bool, help='Evaluation or not')
    # parser.add_argument('--methods', type=str, default='USOD10K', help='evaluated method name')
    # parser.add_argument('--save_dir', type=str, default='./', help='path for saving result.txt')
    # parser.add_argument('--Testing', default=False, type=bool, help='Testing or not')
    # parser.add_argument('--save_test_path_root', default='preds/', type=str, help='save saliency maps path')
    # parser.add_argument('--test_paths', type=str, default='../data/USOD10k/USOD10K-TE')
    # parser.add_argument('--data_root', default='', type=str, help='data path')

    pred_dir = args.save_test_path_root
    output_dir = args.save_dir
    gt_dir = args.data_root

    method_names = args.methods.split('+')

    threads = []
    test_paths = args.test_paths.split('+')
    for dataset_setname in test_paths:

        dataset_name = dataset_setname.split('/')[0]

        for method in method_names:

            pred_dir_all = osp.join(pred_dir, dataset_name, method)
            #pred_dir_all = osp.join(pred_dir, method)
            if dataset_name in ['USOD10K']:
                gt_dir_all = osp.join(osp.join(gt_dir, dataset_setname), 'testset/GT')
            else:
                gt_dir_all = osp.join(osp.join(gt_dir, dataset_setname), 'GT')

            loader = EvalDataset(pred_dir_all, gt_dir_all)
            thread = Eval_thread(loader, method, dataset_setname, output_dir, cuda=True)
            threads.append(thread)
    for thread in threads:
        print(thread.run())

