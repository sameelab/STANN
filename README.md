
<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


```sh

        ▄▄▄▄▄▄▄▄▄▄▄       ▄▄▄▄▄▄▄▄▄▄▄       ▄▄▄▄▄▄▄▄▄▄▄       ▄▄        ▄       ▄▄        ▄ 
       ▐░░░░░░░░░░░▌     ▐░░░░░░░░░░░▌     ▐░░░░░░░░░░░▌     ▐░░▌      ▐░▌     ▐░░▌      ▐░▌
       ▐░█▀▀▀▀▀▀▀▀▀       ▀▀▀▀█░█▀▀▀▀      ▐░█▀▀▀▀▀▀▀█░▌     ▐░▌░▌     ▐░▌     ▐░▌░▌     ▐░▌
       ▐░▌                    ▐░▌          ▐░▌       ▐░▌     ▐░▌▐░▌    ▐░▌     ▐░▌▐░▌    ▐░▌
       ▐░█▄▄▄▄▄▄▄▄▄           ▐░▌          ▐░█▄▄▄▄▄▄▄█░▌     ▐░▌ ▐░▌   ▐░▌     ▐░▌ ▐░▌   ▐░▌
       ▐░░░░░░░░░░░▌          ▐░▌          ▐░░░░░░░░░░░▌     ▐░▌  ▐░▌  ▐░▌     ▐░▌  ▐░▌  ▐░▌
        ▀▀▀▀▀▀▀▀▀█░▌          ▐░▌          ▐░█▀▀▀▀▀▀▀█░▌     ▐░▌   ▐░▌ ▐░▌     ▐░▌   ▐░▌ ▐░▌
                 ▐░▌          ▐░▌          ▐░▌       ▐░▌     ▐░▌    ▐░▌▐░▌     ▐░▌    ▐░▌▐░▌
        ▄▄▄▄▄▄▄▄▄█░▌          ▐░▌          ▐░▌       ▐░▌     ▐░▌     ▐░▐░▌     ▐░▌     ▐░▐░▌
       ▐░░░░░░░░░░░▌          ▐░▌          ▐░▌       ▐░▌     ▐░▌      ▐░░▌     ▐░▌      ▐░░▌
        ▀▀▀▀▀▀▀▀▀▀▀            ▀            ▀         ▀       ▀        ▀▀       ▀        ▀▀ 
                                                                                     
```

<!-- PROJECT LOGO -->

<br />

<p align="center">
  
  <a href="https://github.com/github_username/repo_name">
    
   
  </a>

  <p align="center">
     Given ST and scRNA-seq data of a tissue, STANN models cell-types in the scRNA-seq dataset from the genes that are profiled by both ST and scRNA-seq. The trained STANN model then assigns cell-types to the ST dataset.
    <br />
    <br />
    <br />
    <a href="https://github.com/sameelab/STANN/blob/master/notebooks/demo.ipynb">View Demo</a>
    ·
    <a href="https://github.com/sameelab/STANN/issues">Report Bug</a>
    ·
    <a href="https://github.com/sameelab/STANN/issues">Request Feature</a>
  </p>
</p>


<!-- GETTING STARTED -->

## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

```sh
scanpy==1.4.5.1 anndata==0.7.1 umap==0.3.10 numpy==1.18.1 scipy==1.4.1 
pandas==1.0.1 scikit-learn==0.22.1 statsmodels==0.11.0 python-igraph==0.8.0 
tensorflow==2.1.0 keras==2.2.4-tf
```


<!-- USAGE EXAMPLES -->

## Usage

Install conda environment

```sh
conda env create -f environment.yml
```

Activate conda environment 

```sh
conda activate STANN
```

Run model on your data

```sh
python train.py --model STANN --data_train ./path_to_training_data/data.h5ad --data_predict ./path_to_predict_data/data.h5ad --output ./path_to_output/ --project name_of_project
```

## Demo

Demo data 

<p>
  <p>
    <a href="https://bcm.box.com/s/t0n7pdazg817gw0q81h4djlcsgzhyymz">Download data</a>
  </p>
</p>

Demo code can be run in a jupyter notebook

<p>
  <p>
    <a href="https://github.com/sameelab/STANN/blob/master/notebooks/demo.ipynb">View Demo</a>
  </p>
</p>


<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/sameelab/STANN/issues) for a list of proposed features (and known issues).


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Francisco Grisanti - francisco.grisanticanozo@bcm.edu

Project Link: [https://github.com/sameelab/STANN](https://github.com/sameelab/STANN)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/sameelab/STANN.svg?style=flat-square
[contributors-url]: https://github.com/sameelab/STANN/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/sameelab/STANN.svg?style=flat-square
[forks-url]: https://github.com/sameelab/STANN/network/members
[stars-shield]: https://img.shields.io/github/stars/sameelab/STANN.svg?style=flat-square
[stars-url]: https://github.com/sameelab/STANN/stargazers
[issues-shield]: https://img.shields.io/github/issues/sameelab/STANN.svg?style=flat-square
[issues-url]: https://github.com/sameelab/STANN/issues
[license-shield]: https://img.shields.io/github/license/sameelab/STANN.svg?style=flat-square
[license-url]: https://github.com/sameelab/STANN/blob/master/LICENSE.txt
