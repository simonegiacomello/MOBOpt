# MOBOpt

Multi-Objective Bayesian Optimization

## Prerequisites

  * Python 3.7
  * numpy 1.16
  * matplotlib 3.0
  * scikit-learn 0.22
  * deap 1.3
  * scipy 1.1
  * pyDOE 0.3.8 (Latin Hypercube Sampling)

## Installation

  *  Clone this repo to your local machine using `https://github.com/simonegiacomello/MOBOpt.git`
  *  Run `python3 setup.py install`
  * Using pip `pip3 install https://github.com/simonegiacomello/MOBOpt/archive/master.zip`

## Usage

Check [wiki](https://github.com/ppgaluzio/MOBOpt/wiki) for basic usage and documentation of the original MOBOpt package.

## Analysis

Files `example_smsego.py` and `example_nsga2.py` are used to generate the results with SMS-EGO
and NSGA-II respectively.

File `PrintFront.py` is used to print together the True Pareto Front and the fronts obtained with SMS-EGO and NSGA-II.

File `posterior_plots.py` is used to print prior and posterior distributions of the Gaussian Processes after 10, 30 and 50 iterations of SMS-EGO.
## Cite

To cite the original MOBOpt, please refer to the [paper](https://doi.org/10.1016/j.softx.2020.100520)

```
@article{GALUZIO2020100520,
title = "MOBOpt — multi-objective Bayesian optimization",
journal = "SoftwareX",
volume = "12",
pages = "100520",
year = "2020",
issn = "2352-7110",
doi = "https://doi.org/10.1016/j.softx.2020.100520",
url = "http://www.sciencedirect.com/science/article/pii/S2352711020300911",
author = "Paulo Paneque Galuzio and Emerson Hochsteiner [de Vasconcelos Segundo] and Leandro dos Santos Coelho and Viviana Cocco Mariani"
}
```

For the actual version described in the publication, refer to release [v1.0](https://github.com/ppgaluzio/MOBOpt/releases/tag/v1.0)
