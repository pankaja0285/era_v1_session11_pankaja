import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn.functional as F
from models import resnet

from google.colab.patches import cv2_imshow

class ResNet_CAM(nn.Module):
    def __init__(self, net, layer_k):
        super(ResNet_CAM, self).__init__()
        self.resnet = net
        convs = nn.Sequential(*list(net.children())[:-1])
        self.first_part_conv = convs[:layer_k]
        self.second_part_conv = convs[layer_k:]
        self.linear = nn.Sequential(*list(net.children())[-1:])
        
    def forward(self, x):
        x = self.first_part_conv(x)
        x.register_hook(self.activations_hook)
        x = self.second_part_conv(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = x.view((1, -1))
        x = self.linear(x)
        return x
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.first_part_conv(x)
        

def superimpose_heatmap(heatmap, img, img_norm):
    resized_heatmap = cv2.resize(heatmap.numpy(), (img.shape[2], img.shape[3]))
    resized_heatmap = np.uint8(255 * resized_heatmap)
    resized_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
    superimposed_img = torch.Tensor(cv2.cvtColor(resized_heatmap, cv2.COLOR_BGR2RGB)) * 0.006 + img_norm.permute(1,2,0)
    
    return superimposed_img

def get_grad_cam(net, img, norm):
    net.eval()
    pred = net(img)
    pred[:,pred.argmax(dim=1)].backward()
    gradients = net.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = net.get_activations(img).detach()
    for i in range(activations.size(1)):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)
    
    return torch.Tensor(superimpose_heatmap(heatmap, img, norm).permute(2,0,1))


def load_saved_weights(in_model, save_path, model_pth_file, model_type="resnet18"):
    # Set pth full path
    save_pth_path = f'{save_path}/{model_pth_file}'
    
    if model_type == "resnet18":
        # in_model = resnet.ResNet18() ## create an object of your model
        in_model.load_state_dict(torch.load(save_pth_path))
    curr_model = in_model
    return curr_model
    
def im_show(img):
    # functions to show an image
    fig, ax = plt.subplots(figsize=(12, 12))
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def prepare_image_plot_gradcam(img_loader, inv_norm, load_models={}, model_types=[], 
  save_path="", layer_k=4, n_imgs=10, img_size=64, apply_alb_cutout=""):
    # layer_k = 4
    # n_imgs = 10
    n_rows = 1
    # create model instance
    for model_type in model_types:
        curr_model = None
        curr_model = load_models[model_type]
        if model_type == "resnet18":
            save_weights_file = f"{model_type}.pth"
            curr_model = load_saved_weights(curr_model, save_path, save_weights_file, model_type=model_type)
            baseline_cam_net = ResNet_CAM(curr_model, layer_k)
            n_rows += 1
            break
        # if model_type == "fmix_net":
        #     fmix_cam_net = ResNet_CAM(curr_model, layer_k)
        # if model_type == "mixup_net":
        #     mixup_cam_net = ResNet_CAM(curr_model, layer_k)
        # if model_type == "fmix_plus_net":
        #     fmix_plus_cam_net = ResNet_CAM(curr_model, layer_k)
        
        # if model_type == "vgg19":
        #     baseline_cam_net = ResNet_CAM(vgg19, layer_k)
        # if model_type == "fmix_net":
        #     fmix_cam_net = ResNet_CAM(fmix_net, layer_k)
        # if model_type == "mixup_net":
        #     mixup_cam_net = ResNet_CAM(mixup_net, layer_k)
        # if model_type == "fmix_plus_net":
        #     fmix_plus_cam_net = ResNet_CAM(fmix_plus_net, layer_k)
    
    # Create torch tensor for imgs
    imgs = torch.Tensor(n_rows, n_imgs, 3, img_size, img_size)
    it = iter(img_loader)
    for i in range(0, n_imgs):
        img, _ = next(it)
        curr_inv_norm = inv_norm(img[0])
        imgs[0][i] = curr_inv_norm
        imgs[1][i] = get_grad_cam(baseline_cam_net, img, curr_inv_norm)
        # imgs[2][i] = get_grad_cam(mixup_cam_net, img)
        # imgs[3][i] = get_grad_cam(fmix_cam_net, img)
        # imgs[4][i] = get_grad_cam(fmix_plus_cam_net, img)
    
    if apply_alb_cutout.strip() != "":
        save_img_filename = f"gradcam_at_layer{str(layer_k)}_alb.png"
    else:
        save_img_filename = f"gradcam_at_layer{str(layer_k)}.png"
    torchvision.utils.save_image(imgs.view(-1, 3, img_size, img_size), 
        save_img_filename, nrow=n_imgs, pad_value=1)
    
    # plot the saved image 
    img = cv2.imread(save_img_filename)
    cv2_imshow(img)
    
    # # Instead convert BGR To RGB and plot image with matplotlib.pyplot
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(image)
    # plt.show()
    # Another way
    # im_show(torchvision.utils.make_grid(imgs.view(-1, 3, img_size, img_size)))
    
    
def load_model_show_gradcam(load_models={}, model_types=[], save_path="saved_models", 
  layer=4, num_images=10, img_size=64, apply_alb_cutout=""):
    # Step: create the val loader
    trf_inv_norm = transforms.Normalize((-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010), (1/0.2023, 1/0.1994, 1/0.2010))
    valset = datasets.CIFAR10(root='./data/cifar', 
                          train=False, download=True,
                          transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                            (0.2023, 0.1994, 0.2010)),
                                                        transforms.Resize(size=img_size, antialias=True)]),)
    
    val_loader = data.DataLoader(valset, batch_size=1, shuffle=True, num_workers=8)
    
    # Step: create model instance
    
    # Step: check if model is loaded and exists, then proceed to apply grad_cam
    if not val_loader is None:
        prepare_image_plot_gradcam(val_loader, trf_inv_norm, 
          load_models=load_models, model_types=model_types, save_path=save_path,
          layer_k=layer, n_imgs=num_images, img_size=img_size, apply_alb_cutout=apply_alb_cutout)
    