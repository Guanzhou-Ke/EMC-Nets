# EMC-Nets
The official repos of Efficient Multi-view Clustering Networks


## Abstract
Deep learning has made remarkable progress on multi-view clustering (MvC) in the last decade. Most existing literature 
adopted a broad target to guide network learning, such as minimize the reconstruction loss. It is effective but not 
efficient. In this paper, we proposed a novel framework, named Efficient Multi-view Clustering Networks (EMC-Nets),
 which guarantees the efficiency of the network learning and produces discriminative common representation efficiently 
 in multi- ple sources. Specifically, the proposed method alternates between the instruction process and approximation 
 process during training. The instruction process em- ploys a standard clustering algorithm, such as k-means, 
 to generate pseudo-labels corresponding to the current common representation. The approximation process leverages 
 pseudo-labels to force the network to approximate a reasonable cluster distribution. Experimental results on four 
 real-world datasets demonstrate that the proposed method outperforms state-of-the-art methods.
 

## Architecture

![architecture](./imgs/architecture.png)


## Environment setting

- python 3.7
- pytorch 1.8.1
- CUDA 10.2

We recommend using `Conda` to setup the environment, and run as the following:

1. Create the virtual environment and install the requirements.
    ```
    conda create -n EMC-Nets python=3.7
    conda activate EMC-Nets
    cd EMC-Nets
    conda install --yes --file requirements.txt
    ```
2. Then, use unittest to test this project, following:
    ```
    cd tests
    export PYTHONPATH="../"
    python -m unittest
    ```

## Quickly validation

Run: 
```
python validation.py
```

More training details see `logs/`

## Training

Coming soon.

## Visualization

Here, we present the visualization of BDGP. More details see our paper, please.

![visualization](./imgs/visualization.png)

