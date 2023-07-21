# Optimizing Cross-Domain Music Recommendation with Adaptive User Preference Transfer (AUPT)
The official paper **Optimizing Cross-Domain Music Recommendation with Adaptive User Preference Transfer (AUPT)**, has been accepted by ICDAI-2023. 

In the present study, the researchers put forth a transfer learning approach that utilizes the benefits of meta-networks and the transfer of individualized preferences in order to effectively tackle the difficulty posed by limited data in cross-domain recommendation systems. Historically, embedding and mapping techniques have been employed to incorporate auxiliary data in cross-domain recommendations. However, these methods have been found to have limitations, such as decreased performance when the auxiliary data is not well-aligned with the target domain and a lack of consideration for individual user preferences. To overcome these limitations, the study proposes utilizing the meta-learning paradigm to create a meta-network that can adapt to different target domains and establish personalized connections between them. By training the meta-network on task-oriented optimization, rather than mapping-oriented optimization, the study demonstrates the ability to learn the appropriate initialization of parameters efficiently and effectively for a new task, particularly when dealing with new users. A thorough evaluation of our methodology on a practical dataset showcases its excellence over other leading cross-domain recommendation approaches in terms of recommendation precision. The feasibility of transferring personalized user preferences is gauged by analyzing various models and train/test ratios through the application of the MAE (Mean Absolute Error) and RMSD (Root Mean Squared Deviation) performance evaluation metrics.

## Introduction
This repository provides the implementations of AUPT and three popular baselines (TGT, CMF, EMCDR):
* TGT：Train a MF model with the data of the target domain.
* CMF: [Relational Learning via Collective Matrix Factorization Categories and Subject Descriptors](https://dl.acm.org/doi/pdf/10.1145/1401890.1401969?casa_token=S9kvmlp1bxEAAAAA:v96uHthvspO1ahgCZ1htH8sGl2voMvREqwXVYGf3X4WbvYXaD7tX1OsfXhx4k126HSOOtsbcbf9q) (KDD 2008)
* EMCDR: [Cross-Domain Recommendation: An Embedding and Mapping Approach](https://www.ijcai.org/Proceedings/2017/0343.pdf) (IJCAI 2017)


## Requirements

- Python 3.6
- Pytorch > 1.0
- tensorflow
- Pandas
- Numpy
- Tqdm

## File Structure

```
.
├── code
│   ├── config.json         # Configurations
│   ├── entry.py            # Entry function
│   ├── models.py           # Models based on MF, GMF or Youtube DNN
│   ├── preprocessing.py    # Parsing and Segmentation
│   ├── readme.md
│   └── run.py              # Training and Evaluating 
└── data
    ├── mid                 # Mid data
    │   ├── Digital_Music.csv
    │   ├── CDs_and_Vinyl.csv
    │   └── Musical_Instruments.csv
    ├── raw                 # Raw data
    │   ├── reviews_Digital_Music_5.json.gz
    │   ├── reviews_CDs_and_Vinyl_5.json.gz
    │   └── reviews_Musical_Instruments_5.json.gz
    └── ready               # Ready to use
        ├── _2_8
        ├── _5_5
        └── _8_2
```

## Dataset

We utilized the Amazon Reviews dataset. 
To download the Amazon dataset, you can use the following link: [Amazon Reviews](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews).
Download the three domains: [CDs and Vinyl](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz), [Musical Instruments](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/), [Digital Music](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) (5-scores), and then put the data in `./data/raw`.

You can use the following command to preprocess the dataset. 
The two-phase data preprocessing includes parsing the raw data and segmenting the mid data. 
The final data will be under `./data/ready`.

```python
python entry.py --process_data_mid 1 --process_data_ready 1
```

## Run

Parameter Configuration:

- task: different tasks within `1, 2 or 3`, default for `1`
- base_model: different base models within `MF, GMF or DNN`, default for `MF`
- ratio: train/test ratio within `[0.8, 0.2], [0.5, 0.5] or [0.2, 0.8]`, default for `[0.8, 0.2]`
- epoch: pre-training and CDR mapping training epoches, default for `10`
- seed: random seed, default for `2020`
- gpu: the index of gpu you will use, default for `0`
- lr: learning_rate, default for `0.01`
- model_name: base model for embedding, default for `MF`

You can run this model through:

```powershell
# Run directly with default parameters 
python entry.py

# Reset training epoch to `10`
python entry.py --epoch 20

# Reset several parameters
python entry.py --gpu 1 --lr 0.02

# Reset seed (we use seed in[900, 1000, 10, 2020, 500])
python entry.py --seed 900
```

If you wanna try different `weight decay`, `meta net dimension`, `embedding dimmension` or more tasks, you may change 
the settings in `./code/config.json`. Note that this repository consists of our AUPT and three baselines, TGT, CMF, and EMCDR.