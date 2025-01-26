# Deep Learning Model for Estimating Engagement at Tourist Points of Interest

## About this project

This project develops a deep learning model to estimate the engagement level of tourist Points of Interest (POIs) based on visual features and metadata. The model utilizes a convolutional neural network combined with categorical and textual data to predict engagement. The goal is to optimize content selection, identify patterns in user interaction, and enhance the user experience.

The project began with data preprocessing, including data augmentation techniques to enhance the dataset. The model was then designed using a convolutional neural network, as shown in the diagram below:

![CombinedNN](/.readme_resources/CombinedNN.png)

Next, hyperparameters were optimized using the TPE sampler. The model was trained with Cross Entropy Loss (with label smoothing) as the criterion and Adam as the optimizer, using a learning rate scheduler. Epochs were adjusted, and model was retrained and evaluated, achieving an **accuracy** of **98.408%** on the test set.

## Requirements:

* NVIDIA CUDA: [Installation guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) / [Installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

* NVIDIA cuDNN: [Installation guide](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/index.html)

* PyTorch: [Installation guide](https://pytorch.org/get-started/locally/)

* Environment requirements: see [requirements.txt](https://github.com/luherod/Deep_Learning/blob/main/requirements.txt)

## Files description:

* [Desarrollo_de_un_Modelo_de_Deep_Learning_para_la_Estimaci%C3%B3n_del_Engagement_en_Puntos_de%20Interes_Turisticos.pdf](https://github.com/luherod/Deep_Learning/blob/main/Desarrollo_de_un_Modelo_de_Deep_Learning_para_la_Estimaci%C3%B3n_del_Engagement_en_Puntos_de%20Interes_Turisticos.pdf): Project paper file.

* [POI_engagement_classifier_-_Design_training_evaluation_code.ipynb](https://github.com/luherod/Deep_Learning/blob/main/POI_engagement_classifier_-_Design_training_evaluation_code.ipynb): jupyter notebook with the code of the model development.

* [POI_engagement_classifier_utils.py](https://github.com/luherod/Deep_Learning/blob/main/POI_engagement_classifier_utils.py): Python utility file containing functions used in the notebook.

* [trained_model.pth](https://github.com/luherod/Deep_Learning/blob/main/trained_model.pth): Trained model

* [data_main](https://github.com/luherod/Deep_Learning/tree/main/data_main): folder with the POI images to be imported in the notebook.

* [poi_dataset.csv](https://github.com/luherod/Deep_Learning/blob/main/poi_dataset.csv): CSV file with POI metadata to be imported in the notebook.

* metrics_XXX.pkl, [net_optimization.sqlite3](https://github.com/luherod/Deep_Learning/blob/main/net_optimization.sqlite3): Metrics and data exported from the hyperparameter optimization process.

* [study_log.txt](https://github.com/luherod/Deep_Learning/blob/main/study_log.txt): hyperparameters optimization complete log.

## Author

Lucía Herrero Rodero.
