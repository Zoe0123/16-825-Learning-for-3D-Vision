import numpy as np
import argparse

import torch
from models import cls_model
from utils import create_dir, viz_cls
from data_loader import get_data_loader

import random
import pytorch3d

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model') #model_epoch_0
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--task', type=str, default="cls", help='The task: cls or seg')
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model().to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    # test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    # test_label = torch.from_numpy(np.load(args.test_label))

    # ------ TO DO: Make Prediction ------
    test_dataloader = get_data_loader(args=args, train=False)

    # 3.1 rotation
    # rot = torch.tensor([30,0,0])
    # R = pytorch3d.transforms.euler_angles_to_matrix(rot, 'XYZ')
    # test_dataloader.dataset.data = (R @ test_dataloader.dataset.data.transpose(1, 2)).transpose(1, 2)
    # rad = torch.Tensor([90 * np.pi / 180.])[0]

    # R_x = torch.Tensor([[1, 0, 0],
    #                     [0, torch.cos(rad), - torch.sin(rad)],
    #                     [0, torch.sin(rad), torch.cos(rad)]])
    # R_y = torch.Tensor([[torch.cos(rad), 0, torch.sin(rad)],
    #                     [0, 1, 0],
    #                     [- torch.sin(rad), 0, torch.cos(rad)]])
    # R_z = torch.Tensor([[torch.cos(rad), - torch.sin(rad), 0],
    #                     [torch.sin(rad), torch.cos(rad), 0],
    #                     [0, 0, 1]])

    # test_dataloader.dataset.data = ((R_x @ R_y @ R_z) @ test_dataloader.dataset.data.transpose(1, 2)).transpose(1, 2)


    correct_obj = 0
    num_obj = 0
    preds_labels = []
    for batch in test_dataloader:
        point_clouds, labels = batch
        point_clouds = point_clouds[:, ind].to(args.device)
        labels = labels.to(args.device).to(torch.long)

        with torch.no_grad():
            pred_labels = torch.argmax(model(point_clouds), dim=-1, keepdim=False)
        correct_obj += pred_labels.eq(labels.data).cpu().sum().item()
        num_obj += labels.size()[0]

        preds_labels.append(pred_labels)

    accuracy = correct_obj / num_obj
    print(f"test accuracy: {accuracy}")
    preds_labels = torch.cat(preds_labels).detach().cpu()

    # Visualize a few random test point clouds and failed test point clouds

    fail_inds = torch.argwhere(preds_labels != test_dataloader.dataset.label)
    for i in range(min(15, len(fail_inds))):
        random_ind = random.randint(0, preds_labels.shape[0]-1)
        while random_ind in fail_inds:
            random_ind = random.randint(0, preds_labels.shape[0]-1)
        verts = test_dataloader.dataset.data[random_ind, ind]
        gt_cls = test_dataloader.dataset.label[random_ind].to(torch.long).detach().cpu().data
        pred_cls = preds_labels[random_ind].detach().cpu().data

        path = f"images_q4/1. random_vis_{random_ind}_with_gt_{gt_cls}_pred_{pred_cls}.gif"
        viz_cls(verts, path, "cuda")
    
    for i in range(len(fail_inds)):
        fail_ind = fail_inds[i]
        verts = test_dataloader.dataset.data[fail_ind, ind]
        gt_cls = test_dataloader.dataset.label[fail_ind].detach().cpu().data
        pred_cls = preds_labels[fail_ind].detach().cpu().data
        path = f"images_q4/1.1 fail_vis_{fail_ind}_with_gt_{gt_cls}_pred_{pred_cls}.gif"
        viz_cls(verts, path, "cuda")

    print(f"test accuracy: {accuracy}")
    
    # # Compute Accuracy
    # test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    # print ("test accuracy: {}".format(test_accuracy))