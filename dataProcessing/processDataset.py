#try:
#    import cupy as np
#except ImportError:
#    import numpy as np
import numpy as np
import glob
import os
import argparse
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import json
from yangDistance import yangVectorDistance
from multiprocessing import Pool
import random

def get_min_distance(mixture_sample, component_choice, metric="euclidian"):
    component_vector = np.repeat(component_choice, mixture_sample.shape[0],axis=0)
    #print(mixture_sample.shape, component_vector.shape)
    if metric == "euclidian":
        #distances = np.sqrt(np.sum((mixture_sample - component_vector)**2, axis=1)).flatten()
        distances = np.linalg.norm(mixture_sample - component_vector, axis=1,ord=2)
        # print(distances)
    elif metric == "city block":
        #distances = np.sum(np.abs(mixture_sample - component_vector), axis=1).flatten()
        distances = np.linalg.norm(mixture_sample - component_vector, axis=1,ord=1)
    elif  metric == "yang(p=1)":
        distances = [yangVectorDistance(mixture_sample[i], component_vector, p=1)for i in range(mixture_sample.shape[0])]
    elif  metric == "yang(p=2)":
        distances = [yangVectorDistance(mixture_sample[i], component_vector, p=2)for i in range(mixture_sample.shape[0])]
    else:
        assert False, "{} is an invalid distance metric"
    min_index = np.argmin(distances)
    min_distance = distances[min_index]
    return min_distance, min_index

def single(mixture_sample, component_sample, metric="euclidian", quiet=True):
        distances = np.zeros((1, mixture_sample.shape[0]))
        mixture_remaining_mask = np.ones_like(mixture_sample).astype(bool)
        while mixture_remaining_mask.any():
            iternum = len(mixture_remaining_mask) - mixture_remaining_mask.sum()
            if not quiet and not iternum % 1000:
                print("\t{}/{}".format(iternum, len(mixture_remaining_mask)))
            # randomly sample component instance
            component_choice = component_sample[np.random.choice(range(component_sample.shape[0]), 1)].reshape((1,-1))
            # Find closest remaining mixture sample and record distance
            remaining_mixture_samples = mixture_sample[mixture_remaining_mask]
            if len(remaining_mixture_samples.shape) == 1:
                remaining_mixture_samples = np.expand_dims(remaining_mixture_samples, 1)
            distance, remaining_mixture_index = get_min_distance(remaining_mixture_samples, component_choice, metric=metric)
            distances[0, len(mixture_sample) - mixture_remaining_mask.sum()] = distance
            # Remove mixture sample from above 
            remaining_indices,_ = np.nonzero(mixture_remaining_mask)
            index_of_closest = remaining_indices[remaining_mixture_index]
            mixture_remaining_mask[index_of_closest] = False
        return distances

def makeDistanceCurve(mixture_sample, component_sample, jobs=2, num_curves_to_average=10, metric="euclidian"):
    if jobs == 1:
        distanceVectors = [single(mixture_sample, component_sample, metric) for _  in  range(num_curves_to_average)]
    else:
        p = Pool(processes=jobs)
        distanceVectors = p.starmap(single,[(mixture_sample, component_sample,metric) for _ in range(num_curves_to_average)])
        p.close()
    distances = np.concatenate(distanceVectors,0)
    # Average over the curves
    smoothed_distances = np.mean(distances, axis=0)
    # Calculate [0th,...,99th] percentiles
    distance_curve = np.percentile(smoothed_distances, range(0,100))
    # Clarification: Manuscript says distance curves were normalized to length 1,
    #                but enperiments were all done using L1-Normalization
    distance_curve = distance_curve / distance_curve.sum()
    return distance_curve

def main():
    sample_files = glob.glob(args.sample_directory+"*.json")
    random.shuffle(sample_files)
    features = np.zeros((len(sample_files), 100))
    if args.create_true_values_vector:
        labels = np.zeros((len(sample_files), 1))
    for f_num, sample_path in enumerate(sample_files):
        print("sample {}/{}".format(f_num, len(sample_files)))
        with open(sample_path) as f:
            sample = json.load(f)
            sample["sample"] = np.array(sample["sample"])
            sample["component_assignment"] = np.array(sample["component_assignment"]).astype(bool)
        component_sample = sample["sample"][sample["component_assignment"]].reshape((-1,1))
        mixture_sample = sample["sample"][~sample["component_assignment"]].reshape((-1,1))
        curve = makeDistanceCurve(mixture_sample, component_sample, jobs=args.n_jobs, num_curves_to_average=args.number_curves_to_average, metric=args.distance_metric)
        features[f_num,:] = curve
        if args.create_true_values_vector:
            labels[f_num] = sample["class_prior"]
    np.save(os.path.join(args.sample_directory, "features.npy"), features)
    if args.create_true_values_vector:
        np.save(os.path.join(args.sample_directory, "labels.npy"), labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_directory", type=str, help="directory of samples")
    parser.add_argument("--distance_metric",type=str,
        help="distance metric to use for generating distance curve: {euclidian, city block, yang(p=1), yang(p=2)}",
        default="euclidian")
    parser.add_argument("--number_curves_to_average", type=int,
        help="specify the number of curves to average over when constructing distance curve", default=10)
    parser.add_argument("--n_jobs", type=int, default=2, help="number of jobs to run when computing curves to average over")
    parser.add_argument("--create_true_values_vector", action="store_true", default=False,
        help="set create_true_values_vector flag to true if you wish to evaluate the model's performance\
         on a dataset with a known class prior. \
         This creates a file labels.py that will be used in the estimate script to evaluate the model's performance.")
    args = parser.parse_args()
    main()
