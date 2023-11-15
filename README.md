# Functional Modularity Development in Abstract Visual Reasoning networks

## Abstract

We introduce and execute a framework to study how groups of neural network parameters required to execute specific
tasks (functional modules) develop over the course of training. We study functional module development in neural networks trained on
abstract visual reasoning (AVR) problems, due to the highly compositional nature of these tasks, which we believe increases the
likelihood of functional modules developing in the trained weights. To identify the weights that are specialised to a particular task over
training, we use a recently introduced tool that optimizes a binary mask on the network weights. Using this tool, we are able to locate
weights in a state-of-the-art AVR model that are specialised to the task of perceiving shapes in AVR problems. Further analysis shows that these modules comprise only ten percent of the total weights
in the network and that there is a significant degree of parameter sharing between the functional modules responsible for the
perception of different shapes. We further make a small number of yet unexplained empirical observations about the properties of these functional modules that provide a clear direction for future study.*
<br><br>


## Directory structure - please see the report for further context

`binary_masks` - Contains the trained binary masks that identify the functional modules \
`datasets` - Contains the abstract visual reasoning datasets used to train the Scattering Compositional Learner (SCL) network and the binary masks \
`model_ckpts` - Contains weights of SCL at various points during it's training
`models` - Implementation of SCL, alongside other models used for testing. \
`wandb` - WandB log files \
`data.py` - Classes for loading the abstract visual reasoning datasets \
`mask_analysis.py` - Functions for binary mask analysis \
`mask_SCL.ipynb` - IPython notebook to apply binary weight masking to SCL. \
`masking.py` - Implements the binary weight masking tool \
`report.pdf` - Project report \
`train_SCL.ipynb` - Trains SCL model on abstract visual reasoning dataset. \
`utils.py` - Utility functions 


<br><br>

## Results overview
<br>

### Example AVR problem

![alt text](report_plots/problem_instances/2x2_grid_originals.png)

An example of an AVR problem. In this study we look to identify groups of weights that are responsible for processing shapes (e.g: circles, squares) *and* track the development of these weights over training. <br><br>


## Verification of the weight masking algorithm
![alt text](report_plots/training_curves/SCL_90_trainingcurves_task.png)

Verification of the binary weight masking algorithm. Zero-ablating the weights that don't contribute to shape processing does not affect the performance of the model significantly. However, higher values of the regularisation parameter reduce the accuracy that the masked model can attain.
<br><br>

## Macroscopic module properties over training

![alt text](report_plots/line_plots.png)

We study macroscopic properties of the functional modules, and find that the modules become increasingly sparse and less dispersed over the network as training progresses. See the paper for the definition of sparsity and degree of dispersion ('dispersity')
<br><br>

## Weight sharing between functional modules

![alt text](report_plots/sharing_matrices/SCL_90.png)

The functional modules identified for each shape share a significant fraction of weights with one another, for the most part. This is to be expected - it's likely that there are functions that underlie the processing of *all* shapes.
