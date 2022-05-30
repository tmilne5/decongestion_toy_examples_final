# decongestion_toy_examples_final
Code for training critics to distinguish between 2-D distributions using Minibatch Optimal Ray Selection (MORS). Results from this code appear in my PhD thesis.


## How to run this code ##
* Create a Python virtual environment with Python 3.8 installed.
* Install the necessary Python packages listed in the requirements.txt file (this can be done through pip install -r /path/to/requirements.txt).
* Run critic_trainer.py with a chosen source and target distribution.

### Important arguments for critic_trainer.py
* source and target; which type distribution to use for $\mu$ and $\nu$. Several options are available.
* source_params and target_params; further specifies the distributions $\mu$ and $\nu$. Enter 0 to see the syntax for your chosen type
* save_dir; where the resulting figures will be saved
* ot; if True, uses Minibatch Optimal Ray Sampling (MORS)
* p; determines the cost function for MORS. $c(x,y) = |x-y|^p$
