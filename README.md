## 1938: Variational Inference in Robotics / Variational Inference Exercise 

Install:

- clone repository `git clone https://github.com/csekeb/tum-vir-exercise.git`

- create a virtual environment, for example, `python -m venv [your-fav-name]` or via `conda`
- activate the environment and install the required packages with `pip install -r requirements.txt`

Instructions:

- Bayesian Logistic Regression
    - cd to folder `logistic-regression` and run `python dataset_logistic_generate.py`to generate data
    - run `python run.py` to train and test a model
    - plot results in `plot_logreg_results.ipynb`
    - read document `texts/vi-exercise.pdf` to understand the code and for further instructions w.r.t. homework and exercises
- Variational Auto-Encoders
    - cd to folder `vae` and run `python run.py --config config_vae.yaml`, this will download MNIST data and train a model
    - run `tensorboard --logdir runs/logs/example_vae/lightning_logs`  to check metrics while training the model, if necessary reduce `trainer/max_epochs` a/o increase learning rate `optimizers/lr` in `config_vae.yaml`  
    - read document `texts/vi-exercise.pdf` to understand the code and for further instructions w.r.t. homework and exercises

