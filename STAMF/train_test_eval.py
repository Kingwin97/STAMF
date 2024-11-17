import os
import torch
import Training
import Testing
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--Training', default=True, type=bool, help='Training or not')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:33112', type=str, help='init_method')
    parser.add_argument('--data_root', default='/home/cvpr/mqw/about_rgbp_work/STAMF/', type=str, help='data path')
    parser.add_argument('--train_steps', default=120000, type=int, help='train_steps')
    parser.add_argument('--img_size', default=224, type=int, help='network input size')
    parser.add_argument('--pretrained_model', default='../pretrained_model/80.7_T2T_ViT_t_14.pth.tar', type=str, help='load pretrained model')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
    parser.add_argument('--lr', default=1e-4, type=int, help='learning rate')
    parser.add_argument('--epochs', default=800, type=int, help='epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
    parser.add_argument('--stepvalue1', default=80000, type=int, help='the step 1 for adjusting lr')
    parser.add_argument('--stepvalue2', default=120000, type=int, help='the step 2 for adjusting lr')
    parser.add_argument('--trainset', default='data/USODP/train', type=str, help='Trainging set')
    parser.add_argument('--save_model_dir', default='checkpoint/', type=str, help='save model path')
    # test
    parser.add_argument('--Testing', default=True, type=bool, help='Testing or not')
    parser.add_argument('--save_test_path_root', default='preds/', type=str, help='save saliency maps path')
    parser.add_argument('--test_paths', type=str, default='data/USODP/test')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    num_gpus = torch.cuda.device_count()
    if args.Training:
        Training.train_net(num_gpus=num_gpus, args=args)
    if args.Testing:
        Testing.test_net(args)
