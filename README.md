# HDNet_TF2
This repository is mainly adopted from [yasaminjafarian/HDNet_TikTok](https://github.com/yasaminjafarian/HDNet_TikTok) with the implementation of TensorFlow 2.12 (has been tested with docker container `tensorflow/tensorflow:2.12.0rc0-gpu`)


## Pre-processing steps
Please refer to [instructions](https://github.com/yasaminjafarian/HDNet_TikTok/blob/main/training/README.md) in the origin repo.

## Train

### Sample data preparation
Simply run `bash download_data.sh` to download the sample data.

### Normal Estimator
```bash
cd training_code
python training_NormalEstimator.py
```

### Depth Estimator
```bash
cd training_code
python training_DepthEstimator.py
```

### HDNet
```bash
cd training_code
python training_HDNet.py
```

After above three steps, you will have model checkpoints in `training_progress/model` under the root directory.


## Inference

The sample can be downloaded via `download_data.sh` script.

```
cd testing_code
python HDNet_Inference.py 
```


