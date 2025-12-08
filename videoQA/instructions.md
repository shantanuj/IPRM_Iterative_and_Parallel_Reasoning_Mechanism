# VideoQA Setup Instructions

## Datasets
1. **STAR Dataset**: Download STAR from [https://bobbywu.com/STAR/](https://bobbywu.com/STAR/)
2. **AGQA Dataset**: Download AGQA from [https://cs.stanford.edu/people/ranjaykrishna/agqa/](https://cs.stanford.edu/people/ranjaykrishna/agqa/)
3. **CLEVRER-Humans Dataset**: Download CLEVRER-Humans from [http://clevrer.csail.mit.edu/](http://clevrer.csail.mit.edu/) and [https://github.com/xyang23/CLEVRER-Humans1.0](https://github.com/xyang23/CLEVRER-Humans1.0)

## Base Scene Graph Features
We use symbolic scene graph inputs instead of raw video features. For STAR and AGQA (built on Charades videos), we use STTrans to get predicted scene graphs and bounding boxes. You can also run it on a perfect oracle perception by using the Charades ground truth scene graphs and bounding boxes. See `STAR_Data_preprocessing.ipynb` and `AGQA-Dataset-processing.ipynb` for formatting into appropriate order as well as `ST_Trans_processing` for processing predictions. Similarly, for CLEVRER, you can use the original dataset object detector and attribute features.

## Training

To train a model, run: `python train_main.py --config=configs/star_iprm_m6_t9.yaml`