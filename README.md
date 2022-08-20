# Robust Node Classification on Graphs: Jointly from Bayesian Label Transition and Topology-based Label Propagation (LInDT)

### Authors: Jun Zhuang, Mohammad Al Hasan.

### Paper:
> Accepted for CIKM '22

### Abstract:
> Node classification using Graph Neural Networks (GNNs) has been widely applied in various real-world scenarios. However, in recent years, compelling evidence emerges that the performance of GNN-based node classification may deteriorate substantially by topological perturbation, such as random connections or adversarial attacks. Various solutions, such as topological denoising methods and mechanism design methods, have been proposed to develop robust GNN-based node classifiers but none of these works can fully address the problems related to topological perturbations. Recently, the Bayesian label transition model is proposed to tackle this issue but its slow convergence may lead to inferior performance. In this work, we propose a new label inference model, namely LInDT, which integrates both Bayesian label transition and topology-based label propagation for improving the robustness of GNNs against topological perturbations. LInDT is superior to existing label transition methods as it improves the label prediction of uncertain nodes by utilizing neighborhood-based label propagation leading to better convergence of label inference. Besides, LIndT adopts asymmetric Dirichlet distribution as a prior, which also helps it to improve label inference. Extensive experiments on five graph datasets demonstrate the superiority of LInDT for GNN-based node classification under three scenarios of topological perturbations.

### Dataset:
> cora, citeseer, pubmed, amazoncobuy, coauthor

### Getting Started:
#### Prerequisites
> Linux or macOS \
> CPU or NVIDIA GPU + CUDA CuDNN \
> Python 3 \
> pytorch, dgl, numpy, scipy, sklearn, numba

#### Clone this repo
> ```git clone https://github.com/junzhuang-code/LInDT.git``` \
> ```cd LInDT/src```

#### Install dependencies
> For pip users, please type the command: ```pip install -r requirements.txt``` \
> For Conda users, you may create a new Conda environment using: ```conda env create -f environment.yml```

#### Directories
> **config**: the config files; \
> **src**: source code and demo; \
> **data**: output labels, perturbed graphs, and model weights.

#### Examples
> Run the **LInDT_Demo.ipynb** via **Jupyter Notebook** for demo \
> Train the node classifier on the train graph: ```python train_GNN.py``` \
> Implement {rdmPert, infoSparse, advAttack} perturbations: ```python {rdmPert, infoSparse, advAttack}.py``` \
> Employ LInDT model to infer labels: ```python main.py``` \
> Visualization with Tensorboard: ```Tensorboard --logdir=./runs/Logs_LInDT --port=8999```

### Source of Competing Methods:
> GNN-Jaccard: [Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](https://www.ijcai.org/proceedings/2019/0669.pdf) [[Code](https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/defense/gcn_preprocess.py)] \
> GNN-SVD: [All You Need Is Low (Rank): Defending Against Adversarial Attacks on Graphs](https://dl.acm.org/doi/pdf/10.1145/3336191.3371789) [[Code](https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/defense/gcn_preprocess.py)] \
> DropEdge: [DropEdge: Towards Deep Graph Convolutional Networks on Node Classification](https://openreview.net/pdf?id=Hkx1qkrKPr) [[Code](https://github.com/DropEdge/DropEdge)] \
> GRAND: [Graph Random Neural Network for Semi-Supervised Learning on Graphs](https://arxiv.org/pdf/2005.11079.pdf) [[Code](https://github.com/THUDM/GRAND)] \
> RGCN: [Robust Graph Convolutional Networks Against Adversarial Attacks](https://dl.acm.org/doi/10.1145/3292500.3330851) [[Code](https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/defense/r_gcn.py)] \
> ProGNN: [Graph Structure Learning for Robust Graph Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3394486.3403049) [[Code](https://github.com/ChandlerBang/Pro-GNN)] \
> GDC: [Bayesian Graph Neural Networks with Adaptive Connection Sampling](http://proceedings.mlr.press/v119/hasanzadeh20a/hasanzadeh20a.pdf) [[Code](https://github.com/armanihm/GDC)] \
> MC dropout: [Dropout as a bayesian approximation: Representing model uncertainty in deep learning](https://proceedings.mlr.press/v48/gal16.pdf)

### Cite
Please cite our paper if you think this repo is helpful.
```
@article{zhuang2022robust,
  author={Zhuang, Jun and Al Hasan, Mohammad},
  year={2022},
  month={Aug.},
  title={Robust Node Classification on Graphs: Jointly from Bayesian Label Transition and Topology-based Label Propagation},
  journal={},
  volume={},
  number={},
  pages={},
  DOI={},
  url={}
}
```
