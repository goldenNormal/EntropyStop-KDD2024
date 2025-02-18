# Introduction
Welcome! This is the official code for paper "EntropyStop: Unsupervised Deep Outlier Detection with Loss Entropy", which has been accepted by KDD 2024.

full paper link: https://arxiv.org/abs/2405.12502

# Requirements
This code requires the following:

- Python>=3.7
- PyTorch>=1.12.1
- Numpy>=1.19.2
- Scipy>=1.6.2
- Scikit-learn>=0.24.1
- PyG >= 2.1.0

Basically, the versions of the above libraries are not strict. You can try to install the latest version of these libraries.

All experiments are conducted on a computer with Ubuntu 22.02 OS, AMD Ryzen 9 7950X CPU, 64GB memory, and an RTX 4090 (24GB GPU memory) GPU.

# Improvements and Efficiency Study
To reproduce the results of Improvement study of AE in Section 5.2,

```
python3 ./Exp1/run_AE_batch.py --earlyStop True
python3 ./Exp1/run_AE_batch.py --earlyStop False
python3 ./Exp1/run_RandNet.py
python3 ./Exp1/run_ROBOD.py
```
For each model, results will be generated in csv format.

# Experiments for RQ3
(Note that this experiment stores all the outlier scores of 300 epoch of 47 datasets in the memory. Therefore, The memory should be at least larger than 24GB. If the memory is not enough, our code can be modified appropriately to meet the small memory.)

To reproduce the results of experiments in Section 5.4,

In the first step, the output of a deep model (i.e. the input of UOMS algorithm) requires to be generated. Take AE as an example,
```
python3 ./Exp3/gen_ae_score.py
```
Then, in the path of `./od-predict-score`, a file with more than 3GB size will be generated.

Next, run all UOMS algorithms to select the optimal epoch:

```
python3 ./Exp3/run_all_uoms.py --model AE
```

The AUC, AP, running time of UOMS algorithms will be generated under `./select_res/`.


The model parameter of above cmd can be set to `AE`, `rdp`, `svdd`, `NTL`, `LoeNTL` for the corresponding deep OD model.

# Contact and Cooperation
If you have any message, please contact the first author's email, i.e., hyh957947142@gmail.com.

# Citation
If you find our work useful, please consider citing our paper below. Thank you!
```
@inproceedings{huang2024entropystop,
  title={Entropystop: Unsupervised deep outlier detection with loss entropy},
  author={Huang, Yihong and Zhang, Yuang and Wang, Liping and Zhang, Fan and Lin, Xuemin},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={1143--1154},
  year={2024}
}
```
