'''Train CIFAR10 with PyTorch.'''
import numpy as np
import argparse
import pandas as pd

from utils import plot_metrics, helper  #, train, test, helper
from models import resnet
import torch
import torch.nn as nn
from pprint import pprint
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataloader.load_data import Cifar10DataLoader
from utils.engine import *
from utils.gradcam_viz import *
import time


def show_sample_images():
    config = helper.process_config("./config/config.yaml")
    # Set up experiment details
    # set it to what we want for the current experiment
    config['model_params']['experiment_name'] = 'CiFar_Model_RES18'
    experiment_name_res = config['model_params']['experiment_name']
    config['data_augmentation']['type'] = "CIFAR10Albumentation"

    config['model_params']['model_for'] = 'res'
    config['model_params']['model_name'] = 'CiFar_Model_RES18'
    config['model_params']['save_model'] = 'Y'

    # Step: set up TriggerEngine 
    trigger_training_res = TriggerEngine(config)
    # Step: get dataloaders
    print("\nGet train and test dataloaders..")
    train_loader, test_loader = trigger_training_res.dataloader()

    # Step: visualize sample images
    print("\nVisualizing sample images..")
    # get some random training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    # show images
    helper.imshow(torchvision.utils.make_grid(images[:16]))
        
        
def run_CiFAR_Resnet_GradCAM_process(curr_args):
    # Step: load config yaml
    config = helper.process_config("./config/config.yaml")
    pprint(config)
    
    # Step: device
    use_cuda = torch.cuda.is_available()
    helper.set_seed(config['model_params']['seed'],use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Step: parser args
    lr_s = ""
    resume = ""
    model_type = ""
    img_size = 0
    apply_alb_cutout = "N"
    apply_alb_cutout_size = 0
    
    lr = 0
    if curr_args.lr is not None:
        lr_s = curr_args.lr
        lr = float(lr_s)
    if curr_args.resume is not None:
        resume = curr_args.resume
    if curr_args.model_type:
        model_type = curr_args.model_type
    if curr_args.cam_img_size is not None:
        cam_img_size = curr_args.cam_img_size
        img_size = int(cam_img_size)
    if curr_args.apply_alb_cutout == "Y":
        apply_alb_cutout = curr_args.apply_alb_cutout
        apply_alb_cutout_size = int(curr_args.apply_alb_cutout_size)
        
    model_res = None
    
    # Step: create model instance and get the model summary
    if model_type == "resnet18":
        print(f"\nCreate {model_type} model instance..")
        model_res = resnet.ResNet18().to(device)
        print(f"Model summary for: {model_type}")
        summary(model_res, input_size=(3, 32, 32))

    # Step: set up engine and train
    if not model_res is None:
        # Set up experiment details
        exp_metrics_res = {}
        # set it to what we want for the current experiment
        experiment_name_res = ""
        if apply_alb_cutout == "Y":
            experiment_name_res = 'CiFar_Model_RES18_alb'
        else:
            experiment_name_res = 'CiFar_Model_RES18'
            
        config['model_params']['experiment_name'] = experiment_name_res
        config['data_augmentation']['type'] = "CIFAR10Albumentation"
    
        config['model_params']['model_for'] = 'res'
        config['model_params']['model_name'] = experiment_name_res
        config['model_params']['save_model'] = 'Y'
    
        if apply_alb_cutout == "Y":
            config['data_loader']['type'] = 'Cifar10DataLoader_Alb'
            config['data_augmentation']['args']['cutout_size'] = apply_alb_cutout_size
            
        # Step: set up TriggerEngine 
        trigger_training_res = TriggerEngine(config)
        # Step: get dataloaders
        if apply_alb_cutout == "Y":
            print("\nGet train and test dataloaders with cutout applied via Albumentations..")
        else:
            print("\nGet train and test dataloaders.")
        train_loader, test_loader = trigger_training_res.dataloader()
    
        # # Step: visualize sample images
        # print("\nVisualizing sample images..")
        # # get some random training images
        # dataiter = iter(train_loader)
        # images, labels = next(dataiter)
        # # show images
        # # helper.imshow(torchvision.utils.make_grid(images[:16]))
        # *** ABOVE MOVED TO THE NOTEBOOK
        
        # Step: trigger_training..
        print("\nTrain..")
        (exp_metrics_res[experiment_name_res]) = trigger_training_res.run_experiment(model_res, train_loader, 
                                                                                     test_loader, lrmin=lr)
        # Step: save experiment
        save_path = ""
        if curr_args.save_path is not None:
            save_path = curr_args.save_path
        helper.create_folder(save_path)
        print("\nSave experiment..")
        trigger_training_res.save_experiment(model_res, experiment_name_res, path=save_path)
        
        # Step: save the model weights as pth file
        print("\nSaving weights as pth file..")
        if apply_alb_cutout == "Y":    
            save_pth_filename = "resnet18_alb.pth" # config['model_params']['save_pth_filename']
        else:
            save_pth_filename = "resnet18.pth" # config['model_params']['save_pth_filename']
        save_pth_path = f"./{save_path}/{save_pth_filename}"
        torch.save(model_res.state_dict(), save_pth_path)
        
        # Step: plot metrics
        print("\nPlot metrics..")
        plot_metrics.plot_metrics(exp_metrics_res[experiment_name_res])
    
        # Step: plot misclassified images
        print("\n\nPlot misclassified images..")
        model_res2 = torch.load(f'./{save_path}/{experiment_name_res}.pt')
        model_res2.eval()
        trigger_training_res.wrong_predictions(model_res2,test_loader, 20)
        
        # Step: plot gradCAM images
        print("\n\nPlot gradCAM images..")
        vgg19 = resnet.ResNet18()
        load_models = {
         "resnet18": vgg19
        }
        model_types=['resnet18']
        load_model_show_gradcam(load_models=load_models, model_types=model_types, save_path=save_path, 
            layer=4, num_images=10, img_size=img_size, apply_alb_cutout=apply_alb_cutout)
        
        # Step: model accuracy
        classes = config['data_loader']['classes']
        print(f"\n\nDisplay model accuracy for {len(classes)} classes..")
        helper.class_level_accuracy(model_res2, test_loader, device, classes)
        print("\nDONE!")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--model_type', default="resnet18", type=str, help='model type to train')
    parser.add_argument('--save_path', default="saved_models", type=str, help='path to save model')
    parser.add_argument('--cam_img_size', default="64", type=str, help='Grad-CAM img size')
    parser.add_argument('--apply_alb_cutout', default="N", type=str, help='Apply Albumentation cutout')
    parser.add_argument('--apply_alb_cutout_size', default="0", type=str, help='Apply Albumentation cutout')
    
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    run_CiFAR_Resnet_GradCAM_process(args)
