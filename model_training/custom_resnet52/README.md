# Cartoon Face Detection and Gender Classification using Detectron2

The first goal of this project is to recreate the benchmark ResNet50 model from the 2020 paper [ACFD: Asymmetric Cartoon Face Detection](https://arxiv.org/abs/2007.00899) and achieve a similar mAP scrore on the [iCartoonFace dettrain dataset](https://github.com/luxiangju-PersonAI/iCartoonFace).

The second goal of this project is to fine-tune the ResNet50 model to classify the gender of cartoon characters based on their faces. 

Instructions on how to run train_net.py from the command line: https://github.com/facebookresearch/detectron2/blob/main/GETTING_STARTED.md

References:
- General
    - https://towardsdatascience.com/object-detection-in-6-steps-using-detectron2-705b92575578
    - https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1
    - [Official Tutorial](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
    - https://github.com/facebookresearch/detectron2/blob/main/projects/DeepLab/train_net.py
    - [Deeplab](https://github.com/facebookresearch/detectron2/tree/main/projects/DeepLab)
        - Augmentations
        - Config setting
        - Code framework used here
- Tensorboard logging metrics
    - https://ortegatron.medium.com/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
- COCO Evaluation
    - https://cocodataset.org/#detection-eval
    - https://blog.zenggyu.com/en/post/2018-12-16/an-introduction-to-evaluation-metrics-for-object-detection/
    - https://github.com/facebookresearch/detectron2/blob/main/tools/train_net.py
- VoVNet-v2 Backbone
    - https://github.com/youngwanLEE/vovnet-detectron2
- Training from scratch
    - https://arxiv.org/abs/1811.08883
- Disney datasets
    - https://animationscreencaps.com/
