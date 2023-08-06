# Human-Gaze-Target Detection Transformer

An (unofficial) PyTorch implementation of the paper "[End-to-End Human-Gaze-Target Detection with Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Tu_End-to-End_Human-Gaze-Target_Detection_With_Transformers_CVPR_2022_paper.pdf)".

## How to run
Install dependencies

```bash
# Clone project
git clone https://github.com/francescotonini/human-gaze-target-detection-transformer
cd human-gaze-target-detection-transformer

# Clone submodule
git submodule update --init --recursive

# Create environment
conda create -n human-gaze-target-detection-transformer python=3.9
conda activate human-gaze-target-detection-transformer

# Install requirements
pip install -r requirements.txt
```

Detect auxiliary faces to improve training
```bash
# GazeFollow
python scripts/gazefollow_get_aux_faces.py --dataset_path /path/to/gazefollow --subset train

# VideoAttentionTarget
python scripts/videoattentiontarget_get_aux_faces.py --dataset_path /path/to/videoattentiontarget --subset train
```

(optional) Setup wandb
```bash
cp .env.example .env

# Add token to .env
```

Train model with default configuration
```bash
# GazeFollow
python src/train.py experiment=hgttr_gazefollow

# VideoAttentionTarget
python src/train.py experiment=hgttr_videoattentiontarget model.net_pretraining={URL TO GAZEFOLLOW PRETRAINING}
```

