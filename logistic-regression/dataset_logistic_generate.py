import numpy as np
import os, os.path
import sklearn.datasets



def gen_and_write_linear(filename, seed_rand, n_data, w_in, w_scale, x_scale):

        if w_in is None:
            w = w_scale * np.ones((1,2))
        else:
            w = w_scale * w_in

        np.random.seed(seed_rand)

        X = x_scale*(2*np.random.rand(n_data, 2) -1)
        y = (1/(1+np.exp(-np.matmul(X, w.T)))) > np.random.rand(X.shape[0],1)
        y  = np.array(y, dtype=np.float64).reshape(len(y),1) 

        data = {"inputs": X.reshape((-1, 2)), "targets": y.reshape((-1, 1))}

        np.savez(filename, **data)
        print(f"{n_data} samples written to {filename}")

def gen_and_write_two_moons(filename, seed_rand, n_data):
        X, y = sklearn.datasets.make_moons(n_samples=n_data, shuffle=True, noise=0.25)
        data = {"inputs": X.reshape((-1, 2)), "targets": y.reshape((-1, 1))}
        np.savez(filename, **data)
        print(f"{n_data} samples written to {filename}")

def generate_data_logistic(directory, seed_rand, n_data=1024, w_in=None,  w_scale=1.0,  x_scale=1.0):
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, name in enumerate(["train", "test", "validate"]):
        dir_curr = os.path.join(directory, name)       
        if not os.path.exists(dir_curr):
            os.makedirs(dir_curr)

        filename = os.path.join(dir_curr, name+".npz")
 
        gen_and_write_linear(
            filename,
            n_data=n_data,
            seed_rand=seed_rand + i,
            w_in = w_in,
            w_scale = w_scale,
            x_scale = x_scale
        )

def generate_data_two_moons(directory, seed_rand, n_data=1024):
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, name in enumerate(["train", "test", "validate"]):
        dir_curr = os.path.join(directory, name)       
        if not os.path.exists(dir_curr):
            os.makedirs(dir_curr)

        filename = os.path.join(dir_curr, name+".npz")
 
        gen_and_write_two_moons(
            filename,
            n_data=n_data,
            seed_rand=seed_rand + i,
        )


def main():
   generate_data_logistic("./data_logistic_regression_2d", seed_rand=2022, n_data=1024, w_scale=10) 
   generate_data_two_moons("./data_logistic_two_moons", seed_rand=2022, n_data=1024) 
 
if __name__ == "__main__":
    main()
    
