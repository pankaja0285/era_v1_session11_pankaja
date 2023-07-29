### Purpose: Use Residual block and create a CNN model for training on CiFAR 10 dataset.

## Based on CiFAR 10 dataset
### Basic structure for the model:- 
-   PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
-   Layer1 - <br/>
    X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]  <br/>
    R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]  <br/>
-   Add(X, R1)
-   Layer 2 -
    Conv 3x3 [256k]  <br/>
    MaxPooling2D <br/>
    BN <br/>
    ReLU <br/>
-   Layer 3 -
    X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k] <br/>
    R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k] <br/>
    Add(X, R2) <br/>
    MaxPooling with Kernel Size 4 <br/>
-   FC Layer 
-   SoftMax
Uses One Cycle Policy such that: <br/>
Total Epochs = 24 <br/>
Max at Epoch = 5 <br/>
LRMIN = FIND <br/>
LRMAX = FIND <br/>
NO Annihilation <br/>
Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8) <br/>
Batch size = 512 <br/>
Use ADAM, and CrossEntropyLoss <br/>

### Project Setup:
Clone the project as shown below:-

```bash
$ git clone git@github.com:pankaja0285/era_v1_session8_pankaja.git
$ cd era_v1_session8_pankaja
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
   __model.py<br/>
|__utils
   __dataset.py<br/
   __engine.py<br/>
   __helper.py<br/>
   __plot_metrics.py<br/>
   __test.py<br/>
   __train.py<br/>
|__CiFAR_S10.ipynb<br/>
|__README.md<br/>

**NOTE:** List of libraries required: ***torch*** and ***torchsummary***, ***tqdm*** for progress bar, which are installed using requirements.txt<br/>

One of 2 ways to run any of the notebooks, for instance **S8.ipynb** notebook:<br/>
1. Using Anaconda prompt - Run as an **administrator** start jupyter notebook from the folder ***era_v1_session10_pankaja*** and run it off of your localhost<br/>
**NOTE:** Without Admin privileges, the installs will not be correct and further import libraries will fail. <br/>
```
jupyter notebook
```
2. Upload the notebook folder ***era_v1_session10_pankaja*** to google colab at [colab.google.com](https://colab.research.google.com/) and run it on colab<br/>

### Group Normalization:
**File used: models/model.py, model with Net2 Class**
<p>
Target:
- create a model with Group Normalization as the normalization method

Results:
- Total parameters: 6,573,120
- Train accuracy of 88.14 and test accuracy of 89.38

Analysis:
- To see how the accuracy using residual blocks.
</p>


### Contributing:
For any questions, bug(even typos) and/or features requests do not hesitate to contact me or open an issue!
