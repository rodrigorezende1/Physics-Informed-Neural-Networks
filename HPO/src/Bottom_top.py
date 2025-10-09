import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

# Set DeepXDE default float type
dde.config.set_default_float("float64")

# --- Problem Parameters ---
dim1 = 1.0
dim2 = 1.2
freq = 50.5 * 10**6
c0 = 299792458
mu0 = 4 * np.pi * 10**(-7)
eps0 = ((1 / c0)**2) / mu0
omega = 2 * np.pi * freq

k1 = (omega**2) * eps0 * mu0
k2 = mu0
k3 = 1.0 # Scaling factor for the PDE residual

# Define the PDE
def pde(x, f):
    """
    Helmholtz equation with a cylindrical source J.
    """
    f_xx = dde.grad.hessian(f, x, i=0, j=0)
    f_yy = dde.grad.hessian(f, x, i=1, j=1)
    source_term = 10**3 / (np.pi * 0.02**2) * \
                  (tf.cast(tf.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2) <= 0.02, tf.float64))
    return (-f_xx - f_yy - k1 * f - k2 * source_term) * k3

# Define the domain
geom = dde.geometry.Rectangle([-dim1 / 2, -dim2 / 2], [dim1 / 2, dim2 / 2])

# Boundary conditions are handled by the transform
bc = []
data = dde.data.PDE(geom, pde, bc, num_domain=500, num_test=50)

# Define the neural network output transformation for hard boundary constraint
def transform(x, y):
    """
    Enforces f=0 on the boundaries.
    """
    res = (dim1 / 2 + x[:, 0:1]) * (x[:, 0:1] - dim1 / 2) * \
          (dim2 / 2 + x[:, 1:2]) * (x[:, 1:2] - dim2 / 2)
    return res * y


# <<< MODIFICATION >>>: Setup for running the search multiple times
num_runs = 10
execution_times = []
max_iterations = 10000

# <<< MODIFICATION >>>: Outer loop to run the entire search 10 times
for run_idx in range(num_runs):
    print(f"\n=============================================")
    print(f"========= Starting Search Run {run_idx + 1}/{num_runs} =========")
    print(f"=============================================")

    # <<< MODIFICATION >>>: Set a seed for reproducibility of each run
    # Using the run index ensures each run is different but repeatable
    tf.random.set_seed(run_idx)
    np.random.seed(run_idx)

    # <<< MODIFICATION >>>: Start timer for this specific run
    start_run_time = time.time()
    exit_loops = False  # Reset flag for each new run

    # --- Grid Search Logic ---
    for num_layers in range(2, 8):
        if exit_loops: break
        
        for num_dense_nodes in range(10, 160, 50):
            print(f"\nTesting Model: Layers={num_layers}, Nodes={num_dense_nodes}")
            
            # Define and compile the model
            net = dde.maps.FNN([2] + [num_dense_nodes] * num_layers + [1], "sin", "Glorot uniform")
            net.apply_output_transform(transform)
            model = dde.Model(data, net)
            model.compile("adam", lr=10**(-3))

            # Train the model
            losshistory, _ = model.train(iterations=max_iterations, display_every=int(max_iterations/10))
            train_loss = np.array(losshistory.loss_train).sum(axis=1).ravel()
            test_loss = np.array(losshistory.loss_test).sum(axis=1).ravel()
            
            min_train_loss = train_loss.min() if train_loss.size > 0 else float('inf')
            min_test_loss = test_loss.min() if test_loss.size > 0 else float('inf')

            print(f"  Min Training Loss: {min_train_loss:.6e}")
            print(f"  Min Test Loss: {min_test_loss:.6e}")
            
            # Check for convergence
            if min_train_loss < 5e-5 and min_test_loss < 1e-4:
                print(f"\nConvergence reached with: Layers={num_layers}, Nodes={num_dense_nodes}")
                exit_loops = True
                break
    
    # <<< MODIFICATION >>>: Stop timer and record the elapsed time for the run
    end_run_time = time.time()
    elapsed_time = end_run_time - start_run_time
    execution_times.append(elapsed_time)

    print(f"\n--- Search Run {run_idx + 1} Concluded ---")
    if not exit_loops:
        print("Search completed without reaching the desired convergence criteria.")
    print(f"Time for this run: {elapsed_time:.2f} seconds")


# <<< MODIFICATION >>>: Calculate and display final statistics and plot
print("\n=============================================")
print("========= Final Performance Results =========")
print("=============================================")

times_array = np.array(execution_times)
mean_time = np.mean(times_array)
variance_time = np.var(times_array)
std_dev_time = np.std(times_array)

print(f"Grid search was executed {num_runs} times.\n")
print(f"Individual run times (seconds): {np.round(times_array, 2)}")
print(f"\nMean execution time: {mean_time:.2f} seconds")
print(f"Variance of execution time: {variance_time:.2f} seconds^2")
print(f"Standard Deviation of execution time: {std_dev_time:.2f} seconds")

# --- Generate Box Plot ---
print("\nGenerating box plot of execution times...")

plt.figure(figsize=(8, 6))
plt.boxplot(times_array, patch_artist=True, labels=["Grid Search"]) # Label for this method

# Add titles and labels for clarity
plt.title("Distribution of Grid Search Execution Times (10 Runs)", fontsize=16)
plt.ylabel("Execution Time (seconds)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Display the plot
plt.show()