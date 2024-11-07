# Automatic coarse-to-fine method for cattle body measurement based on improved GCN and 3D parametric model

This repository represents the source code for our paper.

<p align="center">
  <img src="assets/pipeline.png">
</p>

We propose a novel two-stage coarse-to-fine approach for automatic cattle body measurement that combines parametric models with non-parametric representations. Our method begins with a coarse estimation stage, where a parametric model is initially aligned with the point cloud data based on pose priors, shape priors, and detected key points. Going beyond the shape space of the parametric model, the estimated coarse mesh is then refined by an encoder-decoder structured Graph Convolutional Network (GCN) to predict per-vertex non-parametric deformations. More specifically, the point cloud-level global features and vertex-level local features are jointly integrated as the input of downsampling and upsampling GCNs to fully capture non-linear variations of hierarchical meshes. Finally, body measurements are performed on the reconstructed mesh. 

## Demo
<p align="center">
  <img src="assets/BM-GCN_gradio_demo.gif">
</p>

To start with, in command line, run the following to start the gradio user interface:
```
python app.py
```
Please refer to the GIF above for a step-by-step demonstration of the GUI usage.

## TODO List
- [x] Gradio based GUI.
- [x] Preprocessed data.
- [x] Training and inference. 
- [x] Evaluation code. 
- [ ] Online demo on HuggingFace Spaces.

## Installation
The code has been tested with CUDA 11.1, CuDNN 8.9, Python 3.9 and PyTorch 1.9.0; All experiments in our paper were conducted on a single NVIDIA GeForce 3070Ti GPU.
Our default installation method is based on Conda package and environment management:

### Set up conda environment
```
conda create -n bmgcn python=3.9 -y
conda activate bmgcn
```

### Install key packages
```
conda install pytorch==1.9.0 torchvision==0.10.0 pytorch-cuda=11.1 -c pytorch -c nvidia -y
conda install pytorch3d=0.7.2 -c pytorch3d -y
```

### Install pip dependency
```
pip install -r requirements.txt
```

## Data Preparation
Your folder structure should look as follows:
```bash
folder
├── checkpoints
│   └── ...
├── data
│   ├── cattle_coarse
│   ├── cattle_refine
│   ├── smal
│   ├── predefine
│   ├── priors
│   ├── mesh_down_sampling_4.npz
├── dataset
├── example_images
├── logs
├── model
├── smal_model
├── util
```

### Download Data
* Download the [cattle_coarse data](https://drive.google.com/file/d/1VtLs1hEEuX_TSi1JpErkDGZefhj3WQ-k/view?usp=sharing) and put it under the folder `./data/cattle_coarse`.

* Download the [cattle_refine data](https://drive.google.com/file/d/1VtLs1hEEuX_TSi1JpErkDGZefhj3WQ-k/view?usp=sharing) and put it under the foder `./data/cattle_refine`.

### Download SMAL and Priors
* Download the [SMAL](https://github.com/benjiebob/WLDO/tree/master/data/smal) template and the key point index from [index](https://drive.google.com/file/d/1icIjZGeSaVHkl7pMClfTMWKjI2AViS5s/view?usp=sharing) and put the downloaded files under `./data/smal`.

* Download the [predefined](https://drive.google.com/file/d/15Skn5-vlk4o6R-J96ggW4_knG0Pi3QtY/view?usp=sharing) body measurement index and put it under the folder `./data/predefine`.
  
* Download the [precomputed downsampling](https://drive.google.com/file/d/1gZhAnQOADLIEYT1apodChwJh3krfBQGB/view?usp=sharing) for the SMAL body mesh and put the downloaded file under `./data`.

* Download the [pose prior data](https://github.com/benjiebob/SMALify/tree/master/data/priors) and put the downloaded priors folder under `./data/priors`.

### Configurations
All configurations can be found in `./util/config.py`. If desired you can change the paths. 

## Train
### 1. Coarse Estimation Stage
To start with, in command line, run the following to start the first stage: 
```
python main_coarse.py 
```
### 2. Mesh Refinement Stage
Then you can continue to train stage 2 by running:
```
python main_refine.py --output_dir logs/refine --nEpochs 250 --lr 1e-5 --local_feat --gpu_ids 0 --feature_model pointnet2_msg_refine --save_results
```

## Test
We provide the [pretrained model](https://drive.google.com/file/d/1i9sFN8KEMpvjkZDxKnwQpIyJkkAmU9TB/view?usp=sharing). You can download and put it under the folder `./logs/refine/`. 
Test our model on the cattle data by running:
```
python eval.py --output_dir logs/test --resume logs/refine/model_best.pth.tar --gpu_ids 0 --local_feat --save_results
```
Qualitative results can be generated and saved by adding '--save_results' to the command. 

## Contact
Currently, only `.pyc` files are provided. After setting up the environment, please replace all `.py` in the commands with `.pyc` to execute them. The full source code will be made publicly available upon the publication of the paper.

## Contact
For any questions on this project, please feel free to contact Hexiao Lu (luhexiao@cau.edu.cn).

## License
Please see [LICENSE](./LICENSE).
