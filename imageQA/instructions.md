# ImageQA Setup Instructions

## Dataset Download

1. **CLEVR Dataset**: Download CLEVR from [https://cs.stanford.edu/people/jcjohns/clevr/](https://cs.stanford.edu/people/jcjohns/clevr/)
2. **CLEVR-Humans Dataset**: Download from [https://cs.stanford.edu/people/jcjohns/iep/](https://cs.stanford.edu/people/jcjohns/iep/)
3. **GQA Dataset**: Download from [https://cs.stanford.edu/people/dorarad/gqa/about.html](https://cs.stanford.edu/people/dorarad/gqa/about.html)
4. **NLVRv2 and v1 Dataset**: Download from the official NLVRv2 and NLVRv1 repository

## Feature Extraction

### CLEVR Features
- If using spatial grid features, use `preprocess.py` in the preprocessing directory or `extract_clip_feats.py`
- Extract features using the CLEVR preprocessing scripts provided

### GQA Features
- For GQA feature extraction, refer to [VinVL](https://github.com/pzzhang/VinVL)
- Use VinVL to extract object detection features and bounding boxes


## Training

To train a model, run: 
```python train_main.py --config=configs/m6_t9.yaml```

## Visualization
For visualization, refer to `Visualization_framework.ipynb`