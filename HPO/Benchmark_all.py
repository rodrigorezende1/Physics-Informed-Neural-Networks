import os

os.environ['DDE_BACKEND'] = 'tensorflow'

import deepxde as dde
import numpy as np
import tensorflow as tf
import time
import random
import matplotlib.pyplot as plt
import skopt
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

# Suppress a deprecation warning from skopt
np.int = int
dde.config.set_default_float("float64")

# ============================================================================
# 1. SHARED PROBLEM DEFINITION
# ============================================================================
print("--- Defining the Physics Problem (Hard Constraints) ---")

# General parameters
dim1 = 1.0
dim2 = 1.2
freq = 50.5 * 10**6
c0 = 299792458
mu0 = 4 * np.pi * 10**(-7)
eps0 = ((1 / c0)**2) / mu0
omega = 2 * np.pi * freq
k1 = (omega**2) * eps0 * mu0
k2 = mu0
k3 = 1.0

# Define the PDE
def pde(x, f):
    f_xx = dde.grad.hessian(f, x, i=0, j=0)
    f_yy = dde.grad.hessian(f, x, i=1, j=1)
    source_term = 10**3 / (np.pi * 0.02**2) * \
                  (tf.cast(tf.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2) <= 0.02, tf.float64))
    return (-f_xx - f_yy - k1 * f - k2 * source_term) * k3

# Define the network output transform for hard boundary constraints
def transform(x, y):
    res = (dim1 / 2 + x[:, 0:1]) * (x[:, 0:1] - dim1 / 2) * \
          (dim2 / 2 + x[:, 1:2]) * (x[:, 1:2] - dim2 / 2)
    return res * y

# Create shared geometry and data objects once for efficiency
geom = dde.geometry.Rectangle([-dim1 / 2, -dim2 / 2], [dim1 / 2, dim2 / 2])
data = dde.data.PDE(geom, pde, [], num_domain=500, num_test=50)

# The learning rate will start at 1e-3 and decay by 10% every 2000 steps.
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.1
)
# Custom exception to stop surrogate search early
class StopTraining(Exception):
    pass

# ============================================================================
# 2. SEARCH METHOD IMPLEMENTATIONS
# ============================================================================

def run_grid_search(num_runs, max_iterations):
    """Performs a grid search and returns execution times and architectures."""
    print("\n--- Starting Grid Search Method ---")
    execution_times = []
    final_run_architectures = []

    for run_idx in range(num_runs):
        print(f"Grid Search: Starting Run {run_idx + 1}/{num_runs}")
        tf.random.set_seed(run_idx)
        np.random.seed(run_idx)
        start_run_time = time.time()
        exit_loops = False
        
        for num_layers in range(2, 9):
            if exit_loops: break
            for num_dense_nodes in range(50, 160, 50):
                tf.keras.backend.clear_session() # ADDED LINE
                
                if run_idx == num_runs - 1:
                    final_run_architectures.append((num_layers, num_dense_nodes))
                
                print(f"  Testing: L={num_layers}, N={num_dense_nodes}")
                net = dde.maps.FNN([2] + [num_dense_nodes] * num_layers + [1], "sin", "Glorot uniform")
                net.apply_output_transform(transform)
                model = dde.Model(data, net)
                adamax_optimizer = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)
                model.compile(adamax_optimizer)
                losshistory, _ = model.train(iterations=max_iterations, display_every=max_iterations)
                
                train_loss = np.array(losshistory.loss_train).sum(axis=1).ravel()
                test_loss = np.array(losshistory.loss_test).sum(axis=1).ravel()
                min_train_loss = train_loss.min() if train_loss.size > 0 else float('inf')
                min_test_loss = test_loss.min() if test_loss.size > 0 else float('inf')

                if min_train_loss < 1e-5 and min_test_loss < 5e-5:
                    print("  Convergence criteria met. Stopping search for this run.")
                    exit_loops = True
                    break
                    
        execution_times.append(time.time() - start_run_time)
        print(f"Grid Search: Run {run_idx + 1} finished in {execution_times[-1]:.2f} seconds.")
    return execution_times, final_run_architectures

def run_random_search(num_runs, max_iterations):
    """Performs a random search and returns execution times and architectures."""
    print("\n--- Starting Random Search Method ---")
    execution_times = []
    final_run_architectures = []
    models_to_test = 10
    
    for run_idx in range(num_runs):
        print(f"Random Search: Starting Run {run_idx + 1}/{num_runs}")
        random.seed(run_idx)
        tf.random.set_seed(run_idx)
        start_run_time = time.time()
        
        for model_idx in range(models_to_test):
            tf.keras.backend.clear_session() # ADDED LINE

            num_layers = random.randint(2, 8)
            num_dense_nodes = random.randint(50, 150)
            
            if run_idx == num_runs - 1:
                final_run_architectures.append((num_layers, num_dense_nodes))
            
            print(f"  Testing model {model_idx+1}/{models_to_test}: L={num_layers}, N={num_dense_nodes}")
            
            net = dde.maps.FNN([2] + [num_dense_nodes] * num_layers + [1], "sin", "Glorot uniform")
            net.apply_output_transform(transform)
            model = dde.Model(data, net)
            adamax_optimizer = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)
            model.compile(adamax_optimizer)
            losshistory, _ = model.train(iterations=max_iterations, display_every=max_iterations)
            
            train_loss = np.array(losshistory.loss_train).sum(axis=1).ravel()
            test_loss = np.array(losshistory.loss_test).sum(axis=1).ravel()
            min_train_loss = train_loss.min() if train_loss.size > 0 else float('inf')
            min_test_loss = test_loss.min() if test_loss.size > 0 else float('inf')

            if min_train_loss < 1e-5 and min_test_loss < 5e-5:
                print("  Convergence criteria met. Stopping search for this run.")
                break
                
        execution_times.append(time.time() - start_run_time)
        print(f"Random Search: Run {run_idx + 1} finished in {execution_times[-1]:.2f} seconds.")
    return execution_times, final_run_architectures

def run_mixed_search(num_runs, max_iterations):
    """Performs a mixed grid/random search and returns execution times and architectures."""
    print("\n--- Starting Mixed Grid/Random Search Method ---")
    execution_times = []
    final_run_architectures = []
    subranges = [
        {"num_layers_range": (2, 4), "num_nodes_range": (50, 70)},
        {"num_layers_range": (2, 5), "num_nodes_range": (50, 90)},
        {"num_layers_range": (2, 6), "num_nodes_range": (50, 110)},
        {"num_layers_range": (2, 7), "num_nodes_range": (50, 130)},
        {"num_layers_range": (2, 8), "num_nodes_range": (50, 150)},
    ]
    models_per_subrange = 2

    for run_idx in range(num_runs):
        print(f"Mixed Search: Starting Run {run_idx + 1}/{num_runs}")
        random.seed(run_idx)
        tf.random.set_seed(run_idx)
        start_run_time = time.time()
        exit_search = False

        for i, subrange_config in enumerate(subranges):
            if exit_search: break
            print(f"  Entering subrange {i+1}/{len(subranges)}")
            min_layers, max_layers = subrange_config["num_layers_range"]
            min_nodes, max_nodes = subrange_config["num_nodes_range"]
            
            for _ in range(models_per_subrange):
                tf.keras.backend.clear_session() # ADDED LINE

                num_layers = random.randint(min_layers, max_layers)
                num_dense_nodes = random.randint(min_nodes, max_nodes)
                
                if run_idx == num_runs - 1:
                    final_run_architectures.append((num_layers, num_dense_nodes))
                
                print(f"    Testing: L={num_layers}, N={num_dense_nodes}")
                
                net = dde.maps.FNN([2] + [num_dense_nodes] * num_layers + [1], "sin", "Glorot uniform")
                net.apply_output_transform(transform)
                model = dde.Model(data, net)
                adamax_optimizer = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)
                model.compile(adamax_optimizer)
                losshistory, _ = model.train(iterations=max_iterations, display_every=max_iterations)

                train_loss = np.array(losshistory.loss_train).sum(axis=1).ravel()
                test_loss = np.array(losshistory.loss_test).sum(axis=1).ravel()
                min_train_loss = train_loss.min() if train_loss.size > 0 else float('inf')
                min_test_loss = test_loss.min() if test_loss.size > 0 else float('inf')

                if min_train_loss < 1e-5 and min_test_loss < 5e-5:
                    print("  Convergence criteria met. Stopping search for this run.")
                    exit_search = True
                    break
                    
        execution_times.append(time.time() - start_run_time)
        print(f"Mixed Search: Run {run_idx + 1} finished in {execution_times[-1]:.2f} seconds.")
    return execution_times, final_run_architectures

def run_surrogate_search(num_runs, max_iterations):
    """Performs a surrogate-based (GP) search and returns execution times and architectures."""
    print("\n--- Starting Surrogate-based (GP) Search Method ---")
    execution_times = []
    final_run_architectures = []
    n_calls = 11
    
    dimensions = [
        Integer(low=2, high=8, name="num_dense_layers"),
        Integer(low=50, high=150, name="num_dense_nodes"),
    ]
    
    global ITERATION_COUNTER, CURRENT_RUN_ARCHITECTURES
    
    @use_named_args(dimensions=dimensions)
    def fitness(num_dense_layers, num_dense_nodes):
        tf.keras.backend.clear_session() # ADDED LINE
        
        global ITERATION_COUNTER, CURRENT_RUN_ARCHITECTURES
        CURRENT_RUN_ARCHITECTURES.append((num_dense_layers, num_dense_nodes))
        
        print(f"  (Iteration {ITERATION_COUNTER}/{n_calls}) Testing: L={num_dense_layers}, N={num_dense_nodes}")
        net = dde.maps.FNN([2] + [num_dense_nodes] * num_dense_layers + [1], "sin", "Glorot uniform")
        net.apply_output_transform(transform)
        model = dde.Model(data, net)
        adamax_optimizer = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)
        model.compile(adamax_optimizer)
        losshistory, _ = model.train(iterations=max_iterations, display_every=max_iterations)
        
        train_loss = np.array(losshistory.loss_train).sum(axis=1).ravel()
        test_loss = np.array(losshistory.loss_test).sum(axis=1).ravel()
        min_train_loss = train_loss.min() if train_loss.size > 0 else float('inf')
        min_test_loss = test_loss.min() if test_loss.size > 0 else float('inf')
        
        ITERATION_COUNTER += 1
        
        if min_train_loss < 1e-5 and min_test_loss < 5e-5:
            raise StopTraining()
            
        error = min_test_loss
        if np.isnan(error): error = 1e5
        return error

    for run_idx in range(num_runs):
        print(f"Surrogate Search: Starting Run {run_idx + 1}/{num_runs}")
        ITERATION_COUNTER = 1
        CURRENT_RUN_ARCHITECTURES = [] 
        start_run_time = time.time()
        
        try:
            gp_minimize(
                func=fitness,
                dimensions=dimensions,
                acq_func="EI",
                n_calls=n_calls,
                x0=[3, 55],
                random_state=run_idx
            )
        except StopTraining:
            print("  Convergence criteria met. Stopping surrogate search for this run.")
        
        if run_idx == num_runs - 1:
            final_run_architectures = CURRENT_RUN_ARCHITECTURES
            
        execution_times.append(time.time() - start_run_time)
        print(f"Surrogate Search: Run {run_idx + 1} finished in {execution_times[-1]:.2f} seconds.")
    return execution_times, final_run_architectures

# ============================================================================
# 3. MAIN EXECUTION AND PLOTTING
# ============================================================================

if __name__ == "__main__":
    NUM_RUNS = 100
    MAX_ITERATIONS = 20000
    
    # --- Run all search methods ---
    grid_times, grid_architectures = run_grid_search(NUM_RUNS, MAX_ITERATIONS)
    random_times, random_architectures = run_random_search(NUM_RUNS, MAX_ITERATIONS)
    mixed_times, mixed_architectures = run_mixed_search(NUM_RUNS, MAX_ITERATIONS)
    surrogate_times, surrogate_architectures = run_surrogate_search(NUM_RUNS, MAX_ITERATIONS)
    
    # --- Create the combined box plot ---
    all_times = [grid_times, random_times, mixed_times, surrogate_times]
    labels = ['Grid', 'Random', 'Mixed', 'Surrogate (GP)']
    
    print("\n--- Generating Final Comparison Plot ---")
    plt.figure(figsize=(12, 8))
    plt.boxplot(all_times, patch_artist=True, labels=labels)
    plt.title("Comparison of Hyperparameter Search Method Runtimes (Hard Constraints)", fontsize=16)
    plt.ylabel("Execution Time to Find Converged Model (seconds)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # --- Print the architectures tested in the final run ---
    print("\n" + "="*50)
    print("Architectures Tested in Final Run (Layers, Nodes)")
    print("="*50 + "\n")

    print(f"--- Grid Search ({len(grid_architectures)} tested) ---")
    print(grid_architectures)
    print("\n")
    
    print(f"--- Random Search ({len(random_architectures)} tested) ---")
    print(random_architectures)
    print("\n")

    print(f"--- Mixed Search ({len(mixed_architectures)} tested) ---")
    print(mixed_architectures)
    print("\n")

    print(f"--- Surrogate (GP) Search ({len(surrogate_architectures)} tested) ---")
    print(surrogate_architectures)
    print("\n")