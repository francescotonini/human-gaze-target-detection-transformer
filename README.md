# End-to-End Human-Gaze-Target Detection with Transformers

An unofficial PyTorch implementation of the paper "[End-to-End Human-Gaze-Target Detection with Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Tu_End-to-End_Human-Gaze-Target_Detection_With_Transformers_CVPR_2022_paper.pdf)".

## Prerequisites
### Environment and dependencies
We provide a pip requirements file to install all the dependencies.
We recommend using a conda environment to install the dependencies.

```bash
# Clone project and submodules
git clone --recursive https://github.com/francescotonini/human-gaze-target-detection-transformer.git
cd human-gaze-target-detection-transformer

# Create conda environment
conda create -n human-gaze-target-detection-transformer python=3.9
conda activate human-gaze-target-detection-transformer

# Install requirements
pip install -r requirements.txt
```

(optional) Setup wandb
```bash
cp .env.example .env

# Add token to .env
```

### Dataset preprocessing
The code expects that the datasets are placed under the [data/](data/) folder.
You can change this by modifying the `data_dir` parameter in the configuration files.

```bash
cat <<EOT >> configs/local/default.yaml
# @package _global_

paths:
  data_dir: "{PATH TO DATASETS}"
EOT
```

The implementation requires faces annotations ("auxiliary faces", i.e. the ones not annotated by GazeFollow or VideoAttentionTarget).
Therefore, you need run the following script to extract face annotations.

```bash
# GazeFollow
python scripts/gazefollow_get_aux_faces.py --dataset_path /path/to/gazefollow --subset train
python scripts/gazefollow_get_aux_faces.py --dataset_path /path/to/gazefollow --subset test

# VideoAttentionTarget
python scripts/videoattentiontarget_get_aux_faces.py --dataset_path /path/to/videoattentiontarget --subset train
python scripts/videoattentiontarget_get_aux_faces.py --dataset_path /path/to/videoattentiontarget --subset test
```

## Training
We provide configuration to train on GazeFollow and VideoAttentionTarget (see [configs/experiment/](configs/experiment/)).

```bash
# GazeFollow
python src/train.py experiment=hgttr_gazefollow

# VideoAttentionTarget
python src/train.py experiment=hgttr_videoattentiontarget model.net_pretraining={URL TO GAZEFOLLOW PRETRAINING}
```

## Evaluation
The configuration files are also useful when evaluating the model.

```bash
# GazeFollow
python src/eval.py experiment=hgttr_gazefollow ckpt_path={PATH TO CHECKPOINT}

# VideoAttentionTarget
python src/eval.py experiment=hgttr_videoattentiontarget ckpt_path={PATH TO CHECKPOINT}
```

### Checkpoints
We provide model weights for GazeFollow at [this URL](https://mega.nz/file/NdhmDK5a#dJBiGvflEQqbjoDCNnWyPgEhiohq2Rnke2U9jt3H540).

## Acknowledgements
This code is based on [PyTorch Lightning](https://www.lightning.ai/), [Hydra](https://hydra.cc/), and the official DETR implementation.

## Cite us
If you use this code implementation in your research, please cite us:
```bibtext
@InProceedings{tonini2023object,
  title={Object-aware Gaze Target Detection},
  author={Tonini, Francesco and Dall'Asen, Nicola and Beyan, Cigdem and Ricci, Elisa},
  booktitle={The IEEE International Conference on Computer Vision (ICCV)},
  month={Oct},
  year={2023}
}
```

and the original paper:
```bibtext
@inproceedings{tu2022end,
  title={End-to-end human-gaze-target detection with transformers},
  author={Tu, Danyang and Min, Xiongkuo and Duan, Huiyu and Guo, Guodong and Zhai, Guangtao and Shen, Wei},
  booktitle={2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month={June},
  year={2022},
}
```
