# Meeting 1

## Concepts to go over

#### Before
- introduce BC and myself
- Do you know eachother?
- Introduce yourselves: strength, weaknesses and why you (didn't) chose this project
- This is a hard project
- Did you read the description and reccomended reading?
- Any questions beforehand?

#### Goal

Identifying and analysing the properfties in data and algorithms, that make the problem more suited for deep or shallow learning, so that we can exploit them. Subsequently, the question becomes; what can you tell us about the borders and tipping-points for the actual values of these properties?

###### Sub-goals
1. Choose 2 to 4 project setups, with 2 of those as priority/favourites (*I recommend MNIST and Property Inspection Prediction*).
2. Divide tasks according to peoples strengths
3. Set a goal for every other day for the next week
4. Implement the 2 priority-setups first. If any time is left after that: do the others too (if there are any)
5. Analyse results. Why is one method winning from the other?
6. Report and check results with supervisors
7. Try to identify whether we can swap outcomes from one to the other. Do this by ablation study of hyperparameters and input-features, training-data selection or some form of statistical analysis. Also discuss with supervisors what else you can try.
8. If you find some results, we will make a data-generation tool that is using these new-found features and then try to show that there is a tipping point by varying this feature.
9. Analyse results and write report explaining your results.

###### What are examples of these properties?

- data-set size
- specific features
- correlation between features
- redundancie in data
- algorithm-specific hyperparameters
- specific data samples that may be hard
- number of parameters
- linearities vs. non-linearities
- 'deepness' or 'wideness'
- complexity of the task/data
- data-noise (added artificially?)
- regularisations
- cleverly extracted features
- [ADD YOUR IDEA HERE]


###### What are examples to evaluate on?

- performance
- model-size
- data-hunger
- feature extraction efforts
- explainability 
- robustness
- scaleability
- time efficiency
- [ADD YOUR IDEA HERE]

#### How does deep learning work?
- Modularity
- Forward pass, activations and nonlinearities
- Weight matrices and linear algebra
- Loss function (criterion), Optimizer
- Backward pass, backprop, gradients
- Batches, Batch-size vs Learning Rate
- Normalisation (batch and input), losing signal
- Regularisation (L2, dropout)
- FC, CNNs and RNNs and input formats
- Example on [colab](https://colab.research.google.com/drive/1arq7ZpWoO4Xw1od_RbMTHCl5IwIoYXOZ#scrollTo=hz1o8peP2WZo) 
- Cuda/GPU

#### How to run?
- google colab
- own laptop

#### Other 
- Confidentiality agreement
- Questions?

#### Contact
- When to meet?
- Rocket-chat
- Supervisor
- Drive?
- stijn.verdenius@braincreators.com
- +31645743577

#### Additional material to provide
- slides DL course (especially lecture 2 & 3)
- pre-made models in pytorch for the projects we agree on will be put [here](https://github.com/StijnVerdenius/Leren-Beslsissen-Pytorch-Tutorial) 
- this doc
- [Pytorch example tutorial](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) 
- [How to run with cuda/gpu? blogpost](https://medium.com/jovianml/training-deep-neural-networks-on-a-gpu-with-pytorch-11079d89805) to speed-up training (extra)
- [How to install pytorch on colab](https://medium.com/@nrezaeis/pytorch-in-google-colab-640e5d166f13) 



