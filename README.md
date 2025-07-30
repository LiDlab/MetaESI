<h1 align="center">MetaESI</h1>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#folders">Folders</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#dsi-prediction">DSI prediction</a></li>
        <li><a href="#dsi-binding-site-inference">DSI key sequence feature inference</a></li>
      </ul>
    </li>
    <li>
      <a href="#available-data">Available Data</a>
      <ul>
        <li><a href="#gold-standard-dataset-gsd">Gold Standard Dataset (GSD)</a></li>
        <li><a href="#benchmark-dataset">Benchmark Dataset</a></li>
        <li><a href="#predicted-dub-substrate-interaction-dataset-pdsid">Predicted DUB-Substrate Interaction Dataset (PDSID)</a></li>
      </ul>
    </li>
    <li>
      <a href="#License">License</a>
    </li>
    <li>
      <a href="#Contact">Contact</a>
    </li>
  </ol>
</details>


## About The Project
[![MetaESI](https://img.shields.io/github/v/release/Dianke-Li/MetaESI?include_prereleases)](https://github.com/Dianke-Li/MetaESI/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10866136.svg)](https://zenodo.org/records/10866136)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<p align="center">
  <img src="models/Fig1.png" alt="MetaESI v1.0 architecture" width="900">
  <br>
  <b>Figure</b>: MetaESI Overall Architecture
</p>

**MetaESI** is a knowledge-guided interpretable deep learning framework that learns E3-substrate interactions while performing _de novo_ residue-level inference of their binding interfaces. We implemented a two-stage learning strategy for proteome-wide predictions: a meta-learning phase extracts transferable knowledge across multiple tasks, followed by an E3-specific transfer phase that adapts this knowledge to predict interactions for individual E3s. This enabled comprehensive mapping of the E3-substrate interactome with residue-level interface annotations across humans and seven key model organisms, generating the **MetaESI-Atlas**.


## Getting Started
To get a local copy up and running, follow these steps:

### Dependencies
MetaESI is tested to work under Python 3.8.
The required dependencies for MetaESI are  [Pytorch](https://pytorch.org/), [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) and [scikit-learn](http://scikit-learn.org/).
Check [environments.yml](https://github.com/LiDlab/MetaESI/blob/main/environment.yml) for list of needed packages.

MetaESI can run on Linux environments (tested on Ubuntu 18.04)​. We highly recommend installing and running this software on a computer with a discrete NVIDIA graphics card (models that support CUDA). If there is no discrete graphics card, the program can also run on the CPU, but it may require a longer runtime.

### Installation

1. Clone the repository and `cd` into it:
   ```sh
   git clone https://github.com/LiDlab/MetaESI.git
   cd MetaESI
   ```
2. Create and activate the environement with:
   ```sh
   conda env create -f environment.yml
   conda activate MetaESI
   ```
   Or install manually the dependencies:
   ```sh
   conda create -n MetaESI python==3.8
   conda activate MetaESI
   conda install pyg==2.5.2 -c pyg
   conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
   pip install learn2learn==0.2.0
   conda install pandas==1.5.3
   conda install h5py==3.11.0
   conda install biopython==1.78
   ```
3. Download all data:

   Datasets (validation and test) and features for training MetaESI are provided in [MetaESI data(~82M)](https://zenodo.org/records/10467917/files/data.tar.gz?download=1)

   Uncompress `tar.gz` file into the MetaESI directory
   ```sh
   tar -zxvf data.tar.gz -C /path/to/MetaESI
   ```
The time it takes to install the required software for MetaESI on a "normal" desktop computer is no longer than on a professional computer with a discrete graphics card. Setting up Python and the corresponding dependency packages in the Windows 10 system will not take more than 15 minutes. If you need help, please refer to the [link](https://geekflare.com/pytorch-installation-windows-and-linux/).

### Folders
./src contains the implementation for the fivefold cross-validations and independent tests of MetaESI and Baselines.

./preprocessing contains the selection of gold standard dataset and the coding of protein sequence features and similarity matrix.

./explain contains the invoking of PairExplainer, which is used to analyze the explainability of the queried DSI.

./results contains MetaESI prediction results, explainable analysis results, and trained MetaESI model.

## Usage

### DSI prediction
To predict deubiquitinase substrate interaction (DSI) use `run_DSIPredictor.py` script with the following parameters:

* `dub`             str, Uniprot ID of the queried DUB
* `candidate_sub`            str, Uniprot ID of the candidate substrate corresponding to the queried DUB
* `model_location`             str, DSIPredictor model file location

#### DEMO: obtaining the MetaESI score of [USP10-MDM2](https://www.sciencedirect.com/science/article/pii/S2211124722012761)

```sh
python run_DSIPredictor.py --dub Q14694 --candidate_sub Q00987
```
OR
```sh
python run_DSIPredictor.py -d Q14694 -s Q00987
```

#### Output:

```txt
Importing protein sequence features...
100%|███████████████████████████████████████████████████| 20398/20398 [00:10<00:00, 1993.32it/s]
Done.
Importing normalized sequence similarity matrix...
100%|█████████████████████████████████████████████| 3383863/3383863 [00:05<00:00, 598758.94it/s]
Done.
Transferred model and data to GPU
The MetaESI score of Q14694 and Q00987 is 0.9654.
```

Under normal circumstances, MetaESI typically takes around 100 seconds to predict the MetaESI score for a candidate DSI pair.
If you prefer not to utilize the GPU, you can append `--nogpu` at the end of the command.


### DSI key sequence feature inference
To investigate sequence features that suggest associations between DUBs and substrates.

use `run_PairExplainer.py` script with the following parameters:

* `feat_mask_obj`             str, The object of feature mask that will be learned (`dsi` - DSI, `dub` - DUB, `sub` - SUB)
* `dub`             str, Uniprot ID of the queried DUB
* `candidate_sub`            str, Uniprot ID of the candidate substrate corresponding to the queried DUB
* `model_location`             str, DSIPredictor model file location
* `output_location`             str, PairExplainer output file location
* `lr`             float, The learning rate to train PairExplainer
* `epochs`             int, Number of epochs to train PairExplainer
* `log`             bool, Whether or not to print the learning progress of PairExplainer

#### DEMO: obtaining the PairExplainer results of USP10-MDM2

```sh
python run_PairExplainer.py --feat_mask_obj dsi --dub Q14694 --candidate_sub Q00987 --output_location results/importance/
```
OR
```sh
python run_PairExplainer.py -obj dsi -d Q14694 -s Q00987
```

#### Output:

```txt
Importing protein sequence features...
100%|███████████████████████████████████████████████████| 20398/20398 [00:10<00:00, 1940.45it/s]
Done.
Importing normalized sequence similarity matrix...
100%|█████████████████████████████████████████████| 3383863/3383863 [00:05<00:00, 602453.09it/s]
Transferred model and data to GPU
importance this pair of DSI: 100%|████████████████████████| 10000/10000 [03:41<00:00, 45.17it/s]
The explainable result of Q14694 and Q00987 is saved in 'results/importance/Q14694_Q00987.csv'.
```

Under normal circumstances, PairExplainer takes approximately 300 seconds to predict the importance of each position on a candidate DSI pair.

If you prefer not to utilize the GPU, you can append `--nogpu` at the end of the command. However, this is not recommended as retraining PairExplainer would be necessary, which can take around 4 hours.


### Reproduction instructions for five-fold cross-validations and independent tests

If you want to replicate the five-fold cross-validation and independent testing process of MetaESI, please run the `main.py` script in the src folder.
```sh
cd src/
```
AND
```sh
python main_GSD.py && python main_GSD_MetaESI_variant.py && python main_GSD_ML.py
```

## Available Data

* #### [Gold Standard Dataset (GSD)](https://github.com/LiDlab/MetaESI/raw/master/Supplementary%20Tables/Supplementary%20Table%201.xlsx)
MetaESI has established a rigorous gold standard dataset where the positive set is sourced from [UBibroswer 2.0](http://ubibrowser.bio-it.cn/ubibrowser_v3/) and negative set is derived from [BioGRID](https://thebiogrid.org/). We divided GSD into the cross-validation dataset and the independent test dataset in chronological order.

We also provide **Gold Standard Positive Set (GSP) with inferred binding sites**, please [click](https://github.com/LiDlab/MetaESI/raw/master/Supplementary%20Tables/Supplementary%20Table%206.xlsx) to download.

* #### [Benchmark Dataset](https://github.com/LiDlab/MetaESI/tree/master/results/performance/GSD)

To ensure fair comparison, cross-validation dataset and independent test dataset are intersected with the corresponding datasets from [UbiBrowser 2.0](http://ubibrowser.bio-it.cn/ubibrowser_v3/home/download).

Click to download the [cross-validation results](https://github.com/LiDlab/MetaESI/blob/master/results/performance/GSD/GSD_crossval_prob.csv) and the [independent test results](https://github.com/LiDlab/MetaESI/blob/master/results/performance/GSD/GSD_indtest_prob.csv).

* #### [Predicted DUB-Substrate Interaction Dataset (PDSID)](https://github.com/LiDlab/MetaESI/raw/master/Supplementary%20Tables/Supplementary%20Table%204.xlsx)
MetaESI was used to performed a large-scale proteome-wide DSI scanning, resulting in a predicted DUB-substrate interaction dataset (PDSID) with 19,461 predicted interactions between 85 DUBs and 5,151 substrates.

We also provide **PDSID with inferred binding sites**, please [click](https://github.com/LiDlab/MetaESI/raw/master/Supplementary%20Tables/Supplementary%20Table%204.xlsx) to download.

## License

This project is covered under the **Apache 2.0 License**.

## Contact
Dianke Li: diankeli@foxmail.com

Yuan Liu: liuy1219@foxmail.com

Dong Li: lidong.bprc@foxmail.com
