### Purpose: Use Residual block and create a CNN model for training on CiFAR 10 dataset.

## Based on CiFAR 10 dataset
### Basic structure for the model:- 
-   convolution layer - 1
-   Layer1 - <br/>
    X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU  <br/>
    R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X)  <br/>
-   Add(X, R1)
-   Layer 2 -
    Conv 3x3   <br/>
    MaxPooling2D <br/>
    BN <br/>
    ReLU <br/>
-   Layer 3 -
    X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU  <br/>
    R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X)  <br/>
    Add(X, R2) <br/>
    MaxPooling with Kernel Size 4 <br/>
-   Layer 4 -
    X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU  <br/>
    R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X)  <br/>
    Add(X, R2) <br/>
    MaxPooling with Kernel Size 4 <br/>
-   FC Layer 
-   SoftMax
Uses One Cycle Policy such that: <br/>

Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(16, 16) <br/>
Batch size = 512 <br/>
Use SGD, and CrossEntropyLoss <br/>

### Project Setup:
Clone the project as shown below:-

```bash
$ git clone git@github.com:pankaja0285/era_v1_session11_pankaja.git
$ cd era_v1_session11_pankaja
```
About the file structure</br>
|__config
   __config.yaml<br/>
|__data
|__data_analysis
|__data_loader
   __load_data.py<br/>
   __albumentation.py<br/>
|__models
   __resnet.py<br/>
|__utils   
   __engine.py<br/>
   __gradcam_viz.py<br/>
   __helper.py<br/>
   __plot_metrics.py<br/>
   __test.py<br/>
   __train.py<br/>
|__CiFAR_S10.ipynb<br/>
|__README.md<br/>

**NOTE:** List of libraries required: ***torch*** and ***torchsummary***, ***tqdm*** for progress bar, which are installed using requirements.txt<br/>

One of 2 ways to run any of the notebooks, for instance **Submission_CiFAR_S11_GradCam.ipynb** notebook:<br/>
1. Using Anaconda prompt - Run as an **administrator** start jupyter notebook from the folder ***era_v1_session11_pankaja*** and run it off of your localhost<br/>
**NOTE:** Without Admin privileges, the installs will not be correct and further import libraries will fail. <br/>
```
jupyter notebook
```
2. Upload the notebook folder ***era_v1_session11_pankaja*** to google colab at [colab.google.com](https://colab.research.google.com/) and run it on colab<br/>

### In <i>Submission_CiFAR_S11_GradCam.ipynb</i> - Use case With RandomCrop ONLY:
**File used: models/resnet.py**
**
<p>
Target:
- Resnet18 model with grad-CAM applied and with RandomCrop

Results:
- Total parameters: 11,173,962
- Train accuracy of 91.05 and test accuracy of 89.17 

Analysis:
- To see how the grad-cam - heat map of the presence of the object
</p>

### In <i>Submission_CiFAR_S11_GradCam.ipynb</i> - With Cutout applied via Albumentations:
**File used: models/resnet.py**
**
<p>
Target:
- Resnet18 model with grad-CAM applied and with RandomCrop and cutout applied via Albumentations

Results:
- Total parameters: 11,173,962
- Train accuracy of 84.52 and test accuracy of 88.66 

Analysis:
- To see how the grad-cam - heat map of the presence of the object
</p>

<p>
    Grad-CAM images:
    ![Alt grad-cam](C:\PS\ERA_V1\era_v1_session11_pankaja\gradcam_at_layer4_alb.png)
</p>
### Contributing:
For any questions, bug(even typos) and/or features requests do not hesitate to contact me or open an issue!
