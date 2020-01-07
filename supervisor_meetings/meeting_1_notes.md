# Meeting 1

## Introduction

#### Before
- introduce BC and myself
- Do you know each other?
- Introduce yourselves: strength, weaknesses and why you (didn't) chose this project
- This is a hard project, please realise this
- Did you read the description and recommended reading?
- Any questions beforehand?

#### Goal

Identifying and analysing the properties in data and algorithms, that make the problem more suited for deep or shallow learning, so that we can exploit them. Subsequently, the question becomes; what can you tell us about the borders and tipping-points for the actual values of these properties?

###### Sub-goals
1. Choose 2 to 4 project setups, with 2 of those as priority/favourites 
    - I recommend MNIST and Property Inspection Prediction.
2. Divide tasks according to peoples strengths 
    - e.g. 2x data-processing, 2x model implementation, some tasks for when someone finishes early
3. Set a goal for every other day for the next week
    - just to get a grip of what you can do in the time available, nothing is set in stone
4. Implement the 2 priority-setups first. If any time is left after that: do the others too (if there are any)
5. Analyse results. Why is one method winning from the other?
6. Report and discuss results with supervisors
7. Try to identify whether we can swap outcomes from one to the other. 
    - Try to use previous findings.
    - Also, do this by ablation study of hyper-parameters and input-features, training-data selection or some form of statistical analysis. 
    - Additionally, discuss with supervisors what else you can try.
8. If you find some results, we will **try** to make a data-generation tool that is using these new-found features and then **try** to show that there is a tipping point by varying this feature. 
    - There are no guarantees for results here so if you cannot find any, then that is okay.
9. Analyse results and write report explaining them.
10. If time is left, make a deliverable script which tests for a new dataset or algortihm whether we should likely employ shallow or deep learning.

#### Consider the difference between these properties and model evaluation methods

###### What are examples of these properties?

- data-set size
- specific features
- correlation between features
- redundancy in data
- algorithm-specific hyperparameters
- specific data samples that may be hard or easy
- number of parameters
- linearities vs. non-linearities
- 'deepness' or 'wideness'
- complexity of the task/data
- data-noise (added artificially?)
- regularisations
- cleverly extracted features
- patterns of some sort
- **[ADD YOUR IDEA HERE, BE CREATIVE, PART OF THE ASSIGNMENT]**


###### What are examples of model evaluation methods?

- performance-metrics (e.g. accuracy, loss, overfitting)
- model-size
- data-hunger
- feature extraction efforts
- explainability 
- robustness
- scaleability
- time efficiency
- etc...

#### Quick tutorial: How does deep learning work?
- Modularity
- Forward pass, activations and nonlinearities
- Weight matrices and linear algebra
- Loss function (criterion), Optimizer
- Backward pass, backprop, gradients
- Batches, Batch-size vs Learning Rate
- Normalisation (batch and input), losing signal
- Regularisation (L2, dropout)
- FC, CNNs and RNNs 
- input formats and preprocessing
- Example on [colab](https://colab.research.google.com/drive/1arq7ZpWoO4Xw1od_RbMTHCl5IwIoYXOZ) 
- Cuda/GPU

#### Quick tutorial: How to run?
- google colab
- own laptop via virtualenvs (see readme)

#### Other 
- Confidentiality agreement
- Questions?

#### Contact
- When to meet?
- Rocket-chat
- Supervisor
- stijn.verdenius@braincreators.com

#### Additional material to provide
- slides DL course in email (especially lecture 2 & 3)
- pre-made models in pytorch for the projects we agree on will be put [here](https://github.com/StijnVerdenius/Leren-Beslsissen-Pytorch-Tutorial) 
- this doc
- [preprocessing of NLP data](https://towardsdatascience.com/sentiment-analysis-using-lstm-step-by-step-50d074f09948)
- [Pytorch example tutorial](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) 
- [How to run with cuda/gpu? blogpost](https://medium.com/jovianml/training-deep-neural-networks-on-a-gpu-with-pytorch-11079d89805) to speed-up training (extra)
- [How to install pytorch on colab](https://medium.com/@nrezaeis/pytorch-in-google-colab-640e5d166f13) 



