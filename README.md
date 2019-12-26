# Fast Nonparametric Estimation of Class Proportions in the Positive-Unlabeled Classification Setting

An intuitive and fast nonparametric algorithm to estimate class proportions

## Installing

Clone repository

```
git clone git@github.ccs.neu.edu:dzeiberg/ClassPriorEstimation.git
```

Install dependencies

```
cd ClassPriorEstimation
pip install -r requirements.txt
```

## Generate Training Data
An example call to generate training data
```
mkdir rawTrainData
python dataProcessing/generateTrainSamples.py --trainSetSize 100000 --saveDirectory rawTrainData
```

## Pre-Process Data
Given a directory of raw datasets with each dataset represented by a JSON file in the following format:

```
{
	"sample": [x_1, ..., x_N],
	"component_assignment": [s_1, ..., s_N],
	"class_prior": alpha
}
```
where x<sub>i</sub> represents a sample and s<sub>i</sub> represents the positive v. unlabeled assignment,

the directory of datasets can be processed by calling:

```
python processDataset.py --sample_directory data_directory --distance_metric euclidian --number_curves_to_average 10
```
The set of supported distance metrics to be used when constructing the distance curve is: {euclidian, city block, yang(p=1), yang(p=2)}


## Authors

* [**Daniel Zeiberg**](dzeiberg.github.io)

See also the list of [contributors](https://github.ccs.neu.edu/dzeiberg/ClassPriorEstimation/graphs/contributors) who participated in this project.

## Acknowledgments

* [Predrag Radivojac](https://www.ccs.neu.edu/home/radivojac/)
* [Shantanu Jain](https://www.khoury.northeastern.edu/people/jain-shantanu/)
