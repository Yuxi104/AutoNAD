# Automated Neural Architecture Design for Industrial Defect Detection 
This is an official implementation of our paper [**AutoNAD**](https://arxiv.org/abs/2510.06669) [*IEEE/ASME Transactions on Mechatronics*, 2025]


## Environment Setup
To install the required dependencies, run:
```bash
conda create -n autonad python=3.8
conda activate autonad
pip install -r requirements.txt
```

## Data Preparation
The dataset used in this project **need to be download from [here](https://github.com/Yuxi104/AutoNAD/releases/tag/v1)** (The dataset has been preprocessed).

The directory structure is the standard layout as following.
```plaintext
/path/to/dataset/
  SegmentationClass/
    label1.png
    label2.png
  JPEGImages/
    image1.jpg
    image2.jpg
  ImageSets/
    Segmentation/
      val.txt
      train.txt
      test.txt
```

## Supernet Train
Before running the training script, please modify the following critical parameters in `run/train_super.sh` according to your local environment:
```plaintext
1. CUDA_VISIBLE_DEVICES
Purpose: Specify the IDs of GPUs to use for training.

2. --nproc_per_node
Purpose: Define the number of processes to launch per node, which typically matches the number of GPUs used.

3. --data-path
Purpose: Specify the root directory path of your dataset.
```

After modifying the parameters above, start the supernet training with the following commands:
```bash
cd run
bash train_super.sh
```

## Search
Before running the search script, please configure the relevant parameters in `run/evo.sh` following the same pattern as the **supernet training** step.

After modification, start the search with:
```bash
cd run
bash evo.sh
```

## Retrain
```bash
cd run
bash retrain.sh
```

## Test
```
cd run
bash test.sh
```

## Citation
If you found this work helpful, please consider to cite it. Thank you!
```plaintext
@article{liu2025automated,
  title={Automated Neural Architecture Design for Industrial Defect Detection},
  author={Liu, Yuxi and Ma, Yunfeng and Tang, Yi and Liu, Min and Jiang, Shuai and Wang, Yaonan},
  journal={arXiv preprint arXiv:2510.06669},
  year={2025}
}
```


## Acknowledge
Our repository is built on excellent works include [Autoformer](https://github.com/microsoft/Cream/tree/main/AutoFormer), [ACmix](https://github.com/LeapLabTHU/ACmix) and [imgaug](https://github.com/aleju/imgaug).
