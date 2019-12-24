import os
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score
from yangDistance import yangDistributionDifference
import json


def sampleDistributionParameters():
    # Sample Negative Betac Distribution Parameters
    distr_0_alpha = np.random.uniform(2, 100)
    distr_0_beta = distr_0_alpha * np.random.uniform(1, 10)
    # Sample Positive Beta Distribution Parameters
    distr_1_alpha = distr_0_alpha + distr_0_alpha * np.random.beta(0.5, 0.5)
    distr_1_beta = distr_0_beta + distr_0_beta * np.random.beta(0.5, 0.5)
    return distr_0_alpha, distr_0_beta, distr_1_alpha, distr_1_beta

def getSample(distr_0_alpha, distr_0_beta, distr_1_alpha, distr_1_beta):
    MIXTURE_SIZE_LOW, MIXTURE_SIZE_HIGH = 1000, 10000
    POSITIVE_SIZE_LOW, POSITIVE_SIZE_HIGH = 100, 5000
    CLASS_PRIOR_LOW, CLASS_PRIOR_HIGH = 0.01, 1
    # Sample parameters for single instance 
    size_mixture = np.random.randint(MIXTURE_SIZE_LOW, MIXTURE_SIZE_HIGH)
    size_positive = np.random.randint(POSITIVE_SIZE_LOW, POSITIVE_SIZE_HIGH)
    class_prior = np.random.uniform(CLASS_PRIOR_LOW, CLASS_PRIOR_HIGH)
    # Generate Mixture Sample
    mixtureSample = []
    for s in range(int(size_mixture)):
        distributionAssignment = np.random.binomial(1, class_prior)
        if not distributionAssignment:
            sample = np.random.beta(distr_0_alpha, distr_0_beta)
        else:
            sample = np.random.beta(distr_1_alpha, distr_1_beta)
        mixtureSample.append(sample)

    mixtureSample = np.array(mixtureSample).reshape((-1,1))
    positiveSample = np.random.beta(distr_1_alpha, distr_1_beta, size_positive).reshape((-1,1))
    sample = np.concatenate((positiveSample, mixtureSample), axis=0)
    # Positive v. Mixture Component Label
    component_assignment = np.concatenate((np.ones_like(positiveSample), np.zeros_like(mixtureSample)), axis=0)
    return sample, component_assignment, class_prior


def main():
    # Bin Yang 2019 Eq. 7 distance range into 100 bins
    NUMBER_DISTANCE_BINS = 100
    # For each parameter set, generate 10 samples
    NUMBER_SAMPLES_PER_PARAMETER_SET = 10
    MAX_BIN_CAPACITY = args.trainSetSize / (NUMBER_DISTANCE_BINS)
    # Array to hold current capacity of each bin
    bin_capacities = np.zeros(NUMBER_DISTANCE_BINS, dtype=int)
    binEdges = np.arange(0, 1.0, 1.0/NUMBER_DISTANCE_BINS)
    setsGenerated = 0
    countChanged = False
    while setsGenerated < args.trainSetSize:
        if countChanged:
            print(setsGenerated, "instances generated")
            print(bin_capacities)
            countChanged = False
        distr_0_alpha, distr_0_beta, distr_1_alpha, distr_1_beta = sampleDistributionParameters()
        distr_difference = yangDistributionDifference(distr_0_alpha, distr_0_beta, distr_1_alpha, distr_1_beta)
        sampleBin = np.digitize(distr_difference, binEdges) - 1
        if bin_capacities[sampleBin] < MAX_BIN_CAPACITY:
            countChanged = True
            for _ in range(NUMBER_SAMPLES_PER_PARAMETER_SET):
                setsGenerated += 1
                bin_capacities[sampleBin] += 1
                sample, component_assignment, class_prior = getSample(distr_0_alpha,
                    distr_0_beta, distr_1_alpha, distr_1_beta)
                instanceDict = {"sample":sample.tolist(),
                                "component_assignment": component_assignment.tolist(),
                                "class_prior": class_prior,
                                "distr_0_alpha": distr_0_alpha,
                                "distr_0_beta": distr_0_beta,
                                "distr_1_alpha": distr_1_alpha,
                                "distr_1_beta": distr_1_beta}
                with open(os.path.join(args.saveDirectory, "sample_{}.json".format(setsGenerated)), "w") as f:
                    json.dump(instanceDict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainSetSize", type=int, help="number of samples to generate for the training set")
    parser.add_argument("--saveDirectory", type=str, help="directory to which samples should be saved")
    args = parser.parse_args()

    main()