# Multiscale Mutation Clustering Algorithm (M2C)

```bash
Contact: William Poole (wpoole [at] Caltech [dot] edu) - code-related issues/requests
Contact: Brady Bernard (bbernard@systemsbiology.org) - other purposes
Created: March 2016
Copyright 2016, Institute for Systems Biology.
Licensed under the Apache License, Version 2.0
```

Requirements 
-------------
Python v. 2.7, Numpy 1.9.2, Scipy 0.15.1, and Matplotlib 1.4.3 (for plotting examples from workflow.py).


Installation
-------------
If the required libraries are installed, all code will run by putting the files into your working python directory or a directory in your python path. 

File Descriptions
-------------
### M2CWorkflow.py
This file provides an example script running the multiscale clustering algorithm on mutation data from the gene PTEN and plotting the results. This is a good place to start if you wish to use the algorithm in your work.
### M2CFunctions.py 
This file contains custom functions and classes used by the clustering algorithm. Users are not expected to need to call these functions directly unless they wish to directly modify how the M2C algorithm works. 
### MultiscaleMutationClusteringScript.py 
This file contains the wrapper script which runs the multiscale clustering algorithm. Importing this module and running the function M2C will run the clustering algorithm. 
#### M2C Required Inputs: 
- data: raw data (point mutations) of a given gene or sequence 
- noise_estimate: initial estimate the percent of the data which is noise. Recommended value: % 0f the total mutations which are synonymous

#### M2C Parameters: 
- interval: The interval over which teh data is defined. Defaults to [min(data), max(data)] 
- bandwidths: underlying bandwidths to use for the mixture model. For robust results, these should densely cover the range of feature scales expected in the data. The default setting is bandwidths = [2, 3, 4, 5, 6, 8, 10, 12, 14, 18, 22, 26, 30, 38, 46, 54, 62, 78, 94, 110, 126, 158, 190, 222, 254, 318, 382, 446] WARNING: Using a sparse set of bandwidths (for example, powers of 2) could result in the algorithm running slowly or using very large amounts of memory. 
- min_mut_count: The minimum number of mutations in a cluster. This acts as a filter before clusters from different bandwidths are merged. 
- noise_bias: a probability threshold between 0 and 1 for a data point to be assigned to a gaussian cluster instead of to the noise cluster. For example, if noise_bias is .5, the probability of a cluster emitting a data point must be greater than 50% for that point to be assigned to the cluster. 
- print_outs (True or False): whether or not to printout out additional information as the algorithm runs

#### M2C Outputs: A tuple: (final_clusters, final_params, densities) 
- final_clusters: a list L of sublists, C. Each sublist C represents a cluster and contains the raw data points assigned to that cluster. Noise cluster not returned 
- final_params: a list L of tuples (w, m, s, n) representing gaussian clusters where w is the weight of a cluster, m is the mean, s is the standard deviation, and n is the uniform noise weight. 
- densities: a list of numpy arrays, where each array is a KDE of one of the bandwidths passed into the M2C function. 
