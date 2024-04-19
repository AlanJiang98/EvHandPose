This is the official code of "EvHandPose: Event-based 3D Hand Pose Estimation with Sparse Supervision".

# 1. Installation
Clone this repository and install dependencies.
1. create an environment
```
conda env create -f environment.yml
conda activate EvHand_public
```

# 2.Dataset Preparation
Our real-world dataset is from [EvRealHands](https://github.com/marian42/mesh_to_sdf).  
Please download our dataset to your disk. We use ```$data$``` to represent the absolute path to our dataset.

# 3. Training
Our training process consists of three steps. First, we train FlowNet under supervision with the flow loss,
and fix its parameters in the following steps. Then, we train our model under fully supervision to get reasonable 
good parameter initialization. Finally, we train our model with both labeled data and unlabeled data under semi 
supervision. 

We will introduce our training process as follows. Please refer to our [paper](https://arxiv.org/html/2303.02862v3) for more details about our training strategy.

## 1. FlowNet Training
First, set the values in ```configs/train_flow.yml``` as follows:
1. Set the ```exper.output_dir``` to your output flow model path. We use ```$output_flow_model_path$``` to represent the output path of our flow model;
2. Set the ```data.smplx_path``` to your model path;
3. Set the ```data.data_dir``` to ```$data$```.

Second, run the following script:
```
cd ./scripts
python train.py --configs ../configs/train_flow.yml --gpus 1
```

We train 20 epochs in our experiments.

## 2. Fully-Supervised Training
First, set the values in ```configs/train_supervision.yml``` as follows:
1. Set the ```exper.output_dir``` to your output path. We use ```$output_fully_supervision_model_path$``` to represent the output path of our fully supervision model;
2. Set the ```data.smplx_path``` to your model path;
3. Set the ```data.data_dir``` to ```$data$```.

Second, run the following script:
```
cd ./scripts
python train.py --configs ../configs/train_supervision.yml --gpus 1 --flow_model_path $output_flow_model_path$/last.ckpt
```

We train 40 epochs in our experiments.

## 3. Semi-Supervised Training
First, set the values in ```configs/train_semi.yml``` as follows:
1. Set the ```exper.output_dir``` to your output path. We use ```$output_semi_supervision_model_path$``` to represent the output path of our semi supervision model;
2. Set the ```data.smplx_path``` to your model path;
3. Set the ```data.data_dir``` to ```$data$```.

Second, run the following script:
```
cd ./scripts
python train.py --configs ../configs/train_supervision.yml --gpus 1 
--model_path $output_fully_supervision_model_path$/last.ckpt --config_merge ../configs/train_semi.yml
```

We train 40 epochs in our experiments.

# 4. Inference
For inference, we provide quantitative results of MPJPE and PA-MPJPE in ```scripts/eval.py```.

You can also get predicted mesh, 3d joints, image with warped events, .etc by setting the  ```log.save_result=True```
in  ```configs/eval.yml```.

Then, run the following script:
```
cd ./scripts
python eval.py --config_train $output_semi_supervision_model_path$/train.yml --config_test ../configs/eval.yml
--gpu 1 --model_path $output_semi_supervision_model_path$/last.ckpt 
```

The output results will be saved in ```$output_semi_supervision_model_path$/test```.