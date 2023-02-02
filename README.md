# Parameterized interval-valued aggregation functions in classification of data with large number of missing values

## Paper

This repository contains the implementation of algorithm and experiments that were
made in the following paper:
```
U. Bentkowska, M. Mrukowicz, Parameterized interval-valued aggregation functions in classification of data
with large number of missing values, IWIFSGN2022 submitted
```

Please cite us if you use this source code or the results of this project in your research.


## Implemented algorithm
The algorithm was initially proposed in [this paper](https://www.sciencedirect.com/science/article/pii/S0020025519301689).
This repository contains implementation of extension of binary k-NN algorithm, called F.
This is actually the same as in [this paper](https://ieeexplore.ieee.org/abstract/document/9177592),
where it was additionally generalized to multi-class case. The corresponding Github repository is [here](https://github.com/furoDMGroup/Multi-class-classification-problems-for-the-k-NN-in-the-case-of-missing-values).

In this paper new families of parameterized aggregation functions was 
implemented and their influence on binary classification of datasets with missing data was examined.

# Reproduce results

The experiments are **fully reproducible** due to the usage of 
**random seeds in every applicable place**, so unless change those 
seeds manually the received results should be identical between
multiple, separate runs. 

## Download source code
The preffered way to use this source code is to clone this repository:
```
git clone https://github.com/furoDMGroup/IWIFSGN2022
```
## Prerequisites
You should have a Python interpreter installed.
This repository was tested mainly for Python 3.7+.

### Note about scikit-learn private api since 0.24.0 
Since 0.24.0 version of scikit-learn there is no possibility to
import something from private api. Private api in sklearn is not
considered to be stable and could break backward compatibility.
However, in our experiments it was a need to perform non-standard
cross-validation with usage of private api. 
Due to the inconvenience of this approach and to have a more long-term solution
the fork of scikit-learn, with the necessary changes can be found [here](https://github.com/furoDMGroup/scikit-learn).
**This branch of this repository is compatible with this fork**.

**It is recommended to first install and compile the fork of the scikit-learn.**
After that, the other requirements of this repo could be installed using either conda
or pip.
```
conda install --file requirements.txt
```
```
pip install -r requirements.txt
```

## Preparation of datasets

First please specify destination path for datasets in [setup_path.py](reproduce_results/setup_path.py) script, by changing the value of *datasets_path* variable. Make sure that the given path is accessible by standard operating system user or run all scripts in privileged mode. To download datasets, please execute [download_datasets.py](reproduce_results/download_datasets.py) script. This script will download datasets from the UCI repository. Since one dataset is a proptierary RAR archive, it needs additional software to unpack. Installing this kind of software is outside of the purposes of this script, so please unrar this archive by your own in the same directory as in *dataset_path*.

## Reproduce each dataset result

In [reproduce_results](reproduce_results) package there are scripts with names corresponding to each dataset from UCI. To reproduce results please execute those scripts. Results will be saved into spreadsheet files. 

To concatenate all results into one spreadsheet file you can use [concat_datasets.py](reproduce_results/concat_datasets.py).

To sort results, according to the level of missing values and AUC values you could use [sort_result.py](reproduce_results/sort_result.py) script by passing the file name as command line argument:
```
sort_result.py input_file.xlsx
```
The sorted file will be named: *sorted_input_file.xlsx*


To generate other summaries of the received results
you could run [summary.py](reproduce_results/summary.py).
This script will both write computed statistics on screen and save
it to two files.
Example usage:
```
summary.py concatenated_biodeg.xlsx
```

## Author

If you have any issues with source code or have any question please contact with:
[Marcin Mrukowicz](https://github.com/MarcinMrukowicz).

## License
This project is licensed under BSD3 Licence - see the [LICENSE](LICENSE) file for details.
