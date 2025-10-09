"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""

import deepxde as dde
from matplotlib import pyplot as plt
import numpy as np
import skopt
import tensorflow as tf
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
import time

np.int = int
dde.config.set_default_float("float64")

# General parameters
dim1=1
dim2=1.2
freq=50.5*10**6
c0 = 299792458
mu0 = 4*np.pi*10**(-7)
eps0 = ((1/c0)**2)/mu0
omega = 2*np.pi*freq

k1 = (omega**2)*eps0*mu0
k2 = mu0 
k3=1#10**5
iterations = 10000

def pde(x, f):
    """
    Helmholtz equation with Cylinder J = 1 source.
    """
    f_xx = dde.grad.hessian(f, x, i=0, j=0)  # Second derivative w.r.t. x
    f_yy = dde.grad.hessian(f, x, i=1, j=1)  # Second derivative w.r.t. y
    # Approximation for Cylinder J = 1
    delta = 10**3/(np.pi*0.02**2)*(tf.cast(tf.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2) <= 0.02, tf.float64))
    return (-f_xx - f_yy - k1 * f - k2 * delta)*k3


def transform(x, y):
    res = (dim1/2 + x[:, 0:1]) * (x[:, 0:1]-dim1/2) * (dim2/2 + x[:, 1:2]) * (x[:, 1:2]-dim2/2) #+5*10**(-3)
    return res * y


def boundary(_, on_boundary):
    return on_boundary


def create_model(config):
    num_dense_layers, num_dense_nodes = config

    geom = dde.geometry.Rectangle([-dim1/2, -dim2/2], [dim1/2, dim2/2])


    data = dde.data.PDE(
        geom,
        pde,
        [],
        num_domain=500,
        num_boundary=0,
        num_test=50,
    )

    net = dde.maps.FNN(
        [2] + [num_dense_nodes] * num_dense_layers + [1],
        "sin",
        "Glorot uniform",
    )

    net.apply_output_transform(transform)

    model = dde.Model(data, net)
    model.compile("adam", lr=10**(-3))
    return model


def train_model(model, config):
    losshistory, train_state = model.train(iterations=iterations)
    train = np.array(losshistory.loss_train).sum(axis=1).ravel()
    test = np.array(losshistory.loss_test).sum(axis=1).ravel()
    metric = np.array(losshistory.metrics_test).sum(axis=1).ravel()

    error = test.min()
    return error


# HPO setting
n_calls = 15
#dim_learning_rate = Real(low=1e-4, high=5e-2, name="learning_rate", prior="log-uniform")
dim_num_dense_layers = Integer(low=2, high=8, name="num_dense_layers")
dim_num_dense_nodes = Integer(low=10, high=150, name="num_dense_nodes")
#dim_activation = Categorical(categories=["sin", "sigmoid", "tanh"], name="activation")

dimensions = [
    #dim_learning_rate,
    dim_num_dense_layers,
    dim_num_dense_nodes,
    #dim_activation,
]

default_parameters = [3, 50]


@use_named_args(dimensions=dimensions)
def fitness(num_dense_layers, num_dense_nodes):
    config = [num_dense_layers, num_dense_nodes]
    global ITERATION

    print(ITERATION, "it number")
    # Print the hyper-parameters.
    #print("learning rate: {0:.1e}".format(learning_rate))
    print("num_dense_layers:", num_dense_layers)
    print("num_dense_nodes:", num_dense_nodes)
    #print("activation:", activation)
    print()

    # Create the neural network with these hyper-parameters.
    model = create_model(config)
    # possibility to change where we save
    error = train_model(model, config)
    # print(accuracy, 'accuracy is')

    if np.isnan(error):
        error = 10**5

    ITERATION += 1
    return error


ITERATION = 0


start_time = time.time()

search_result = gp_minimize(
    func=fitness,
    dimensions=dimensions,
    acq_func="EI",  # Expected Improvement.
    n_calls=n_calls,
    x0=default_parameters,
    random_state=1234,
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")


print(search_result.x)

plot_convergence(search_result)
plot_objective(search_result, show_points=True, size=3.8)
plt.show()
