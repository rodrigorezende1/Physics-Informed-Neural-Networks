import deepxde as dde
import numpy as np
import tensorflow as tf
import time
import random # Import the random module
import matplotlib.pyplot as plt
# Set DeepXDE default float type
dde.config.set_default_float("float64")

# --- Problem Parameters (from your original code) ---
dim1 = 1
dim2 = 1.2
freq = 50.5 * 10**6
c0 = 299792458
mu0 = 4 * np.pi * 10**(-7)
eps0 = ((1 / c0)**2) / mu0
omega = 2 * np.pi * freq

k1 = (omega**2) * eps0 * mu0
k2 = mu0
k3 = 1 # Scaling factor for the PDE residual
# hard_constraint = True # Not directly used as the transform handles it

# Define the PDE
def pde(x, f):
    """
    Helmholtz equation with Cylinder J = 1 source.
    """
    f_xx = dde.grad.hessian(f, x, i=0, j=0)  # Second derivative w.r.t. x
    f_yy = dde.grad.hessian(f, x, i=1, j=1)  # Second derivative w.r.t. y
    # Approximation for Cylinder J = 1
    # Note: Using tf.cast and tf.sqrt for compatibility with TensorFlow operations
    delta = 10**3 / (np.pi * 0.02**2) * \
            (tf.cast(tf.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2) <= 0.02, tf.float64))
    return (-f_xx - f_yy - k1 * f - k2 * delta) * k3

# Define the domain
geom = dde.geometry.Rectangle([-dim1 / 2, -dim2 / 2], [dim1 / 2, dim2 / 2])

# Define the boundary condition (Dirichlet, f=0 on boundary)
# Not explicitly needed as the output transform handles it as a hard constraint
# def boundary(x, on_boundary):
#     return on_boundary
bc = [] # Empty list as BC is handled by transform

# Set the PDE problem
data = dde.data.PDE(geom, pde, bc, num_domain=500, num_boundary=0, num_test=50)

# Define the neural network output transformation for hard boundary constraint
def transform(x, y):
    # This transform enforces f=0 on the boundaries x = +/- dim1/2 and y = +/- dim2/2
    res = (dim1 / 2 + x[:, 0:1]) * (x[:, 0:1] - dim1 / 2) * \
          (dim2 / 2 + x[:, 1:2]) * (x[:, 1:2] - dim2 / 2)
    return res * y

# --- Hyperparameter Search Setup ---
# Define the subranges for layers and nodes
# (min_layers, max_layers), (min_nodes, max_nodes)
subranges = [
    {"num_layers_range": (2, 8), "num_nodes_range": (50, 160)},
  #  {"num_layers_range": (2, 4), "num_nodes_range": (50, 70)},
   # {"num_layers_range": (2, 5), "num_nodes_range": (50, 90)},
    #{"num_layers_range": (2, 6), "num_nodes_range": (50, 110)},
    #{"num_layers_range": (2, 7), "num_nodes_range": (50, 130)},
    #{"num_layers_range": (2, 8), "num_nodes_range": (50, 160)},
]

models_per_subrange = 20 # Number of random models to test in each subrange
max_iterations = 10000  # Max training iterations for each model

# --- Main Search Loop ---
start_time = time.time()
exit_search = False  # Flag to exit all loops

for i, subrange_config in enumerate(subranges):
    print(f"\n--- Entering Subrange {i+1} ---")
    min_layers, max_layers = subrange_config["num_layers_range"]
    min_nodes, max_nodes = subrange_config["num_nodes_range"]

    print(f"Current Layer Range: {min_layers}-{max_layers}")
    print(f"Current Node Range: {min_nodes}-{max_nodes}")

    for model_idx in range(models_per_subrange):
        if exit_search:
            break

        # Randomly choose parameters within the current subrange
        num_layers = random.randint(min_layers, max_layers)
        num_dense_nodes = random.randint(min_nodes, max_nodes)

        print(f"\nTesting Model {model_idx + 1} in Subrange {i+1}:")
        print(f"  Number of layers: {num_layers}")
        print(f"  Number of nodes per layer: {num_dense_nodes}")

        # Define the neural network
        net = dde.maps.FNN([2] + [num_dense_nodes] * num_layers + [1], "sin", "Glorot uniform")
        net.apply_output_transform(transform)

        # Compile the model
        model = dde.Model(data, net)
        model.compile("adam", lr=10**(-3))

        # Train the model
        losshistory, train_state = model.train(iterations=max_iterations, display_every=max_iterations/10)

        # Evaluate convergence
        train_loss = np.array(losshistory.loss_train).sum(axis=1).ravel()
        test_loss = np.array(losshistory.loss_test).sum(axis=1).ravel()

        min_train_loss = train_loss.min()
        min_test_loss = test_loss.min()

        print(f"  Min Training Loss: {min_train_loss:.6e}")
        print(f"  Min Test Loss: {min_test_loss:.6e}")

        # Check for convergence
        if min_train_loss < 1*10**(-5) and min_test_loss < 1*10**(-4):
            print(f"\nConvergence reached with: ")
            print(f"  Layers: {num_layers}, Nodes: {num_dense_nodes}")
            exit_search = True  # Set the flag to exit all loops
            break  # Exit the inner loop (models in subrange)

    if exit_search:
        break # Exit the outer loop (subranges)

end_time = time.time()
elapsed_time_full = end_time - start_time
print(f"\nTotal elapsed time for hyperparameter search: {elapsed_time_full:.2f} seconds")

if not exit_search:
    print("\nSearch completed without reaching the desired convergence criteria.")
    print("Consider adjusting subranges, increasing models_per_subrange, or max_iterations.")


#############################################################################################
##################################Computing the model again##################################
#############################################################################################
#parameters = [6, 60]

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
hard_constraint = True

dde.config.set_default_float("float64")
# Define the PDE
def pde(x, f):
    """
    Helmholtz equation with Cylinder J = 1 source.
    """
    f_xx = dde.grad.hessian(f, x, i=0, j=0)  # Second derivative w.r.t. x
    f_yy = dde.grad.hessian(f, x, i=1, j=1)  # Second derivative w.r.t. y
    # Approximation for Cylinder J = 1
    delta = 10**3/(np.pi*0.02**2)*(tf.cast(tf.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2) <= 0.02, tf.float64))
    return (-f_xx - f_yy - k1 * f - k2 * delta)*k3

# Define the domain
geom = dde.geometry.Rectangle([-dim1/2, -dim2/2], [dim1/2, dim2/2])

# Define the boundary condition: f(x, y) = 0 on the boundary
def boundary(x, on_boundary):
    return on_boundary

# Set the PDE problem
bc = [] #hard constraints
data = dde.data.PDE(geom, pde, bc, num_domain=500, num_boundary=0,num_test=50)

# Define the neural network
net = dde.maps.FNN([2] + [parameters[1]] * parameters[0] + [1], "sin", "Glorot uniform")

def transform(x, y):
    res = (dim1/2 + x[:, 0:1]) * (x[:, 0:1]-dim1/2) * (dim2/2 + x[:, 1:2]) * (x[:, 1:2]-dim2/2) #+5*10**(-3)
    return res * y

if hard_constraint == True:
    net.apply_output_transform(transform)
# Define the model
model = dde.Model(data, net)

# Compile the model
start_time = time.time()
exit_loops = False  # Flag to exit both loops
for i in range(0,4):
    model.compile("adam", lr=10**-(3+i))

    # Train the model
    losshistory, train_state = model.train(iterations=10000)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

# Create a grid for visualization
X, Y = np.meshgrid(np.linspace(-dim1/2, dim1/2, 101), np.linspace(-dim2/2, dim2/2, 121))
xy = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

# Predict the solution
f_pred = model.predict(xy)#/3400

# Reshape for plotting
f_pred = f_pred.reshape(121, 101)

# Plot the contour
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, f_pred, levels=50, cmap="viridis")
plt.colorbar(label="f(x, y)")
plt.title("PINNs Solution")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


######
# Assuming f_pred is the solution computed on a 2D grid of shape (101, 101)
dx = (0.5 - (-0.5)) / 100  # Grid spacing in x (uniform grid)
dy = (0.6 - (-0.6)) / 120  # Grid spacing in y (uniform grid)

# Compute partial derivatives using finite differences
#df_dy, df_dx = np.gradient(f_pred, dy, dx)  # df/dy and df/dx
df_dy, df_dx = np.gradient(f_pred, dy, dx)  # df/dy and df/dx

# Curl components (z-component is zero for this case)
curl_x = df_dy/k2  # i-component
curl_y = -df_dx/k2  # -j-component

#curl_x = np.max(curl_x)  # i-component
#curl_y = np.max(curl_y)  # -j-component

# Compute the magnitude of the curl vectors

#Better plot0
magnitude = np.sqrt(curl_x**2 + curl_y**2)  # Original magnitude
curl_x_normalized = curl_x / (magnitude + 1e-10)  # Normalize x-component
curl_y_normalized = curl_y / (magnitude + 1e-10)  # Normalize y-component

# Scale the normalized vectors to a fixed length (e.g., 1)
uniform_length = 1
curl_x_uniform = curl_x_normalized * uniform_length
curl_y_uniform = curl_y_normalized * uniform_length

# Downsample the grid by slicing
step = 2  # Plot every 5th arrow
X_downsampled = X[::step, ::step]
Y_downsampled = Y[::step, ::step]
curl_x_uniform = curl_x_uniform[::step, ::step]
curl_y_uniform = curl_y_uniform[::step, ::step]
magnitude_step = magnitude[::step, ::step]

#######
###H CST
#####
data = np.loadtxt("C:/Users/TET1/Desktop/theorieund PinnA/Data_H_field_v2.0.txt") ## import data
#Data
x_Hx = data[:,0]
x_Hy = data[:,1]
#H_theta = np.zeros([data[:,2].size,2])
H_theta_x = data[:,2]
H_theta_y = data[:,3]


x_Hx_GP = data1[:,0]
x_Hy_GP = data1[:,2]
#H_theta = np.zeros([data[:,2].size,2])
H_theta_x_GP = data1[:,1]
H_theta_y_GP = data1[:,3]


curl_x[85, :] = curl_x[85, :]/(max(-curl_x[85, :])/max(-H_theta_x))
curl_y[85, :] = curl_y[85, :]/(max(-curl_y[85, :])/max(-H_theta_y))


# Pinns_A_Random = np.column_stack((curl_x[85, :], curl_y[85, :]))
# np.savetxt('Pinns_A_Random', Pinns_A_Random, fmt='%f', delimiter=',')


y_maxHx=max(curl_x[85, :-1])
y_minHx=min(curl_x[85, :-1])
y_maxHy=max(curl_y[85, :-1])
y_minHy=min(curl_y[85, :-1])

ErrorHx = sum((curl_x[85, :-1]-H_theta_x)**2)/(H_theta_x.shape[0]*(y_maxHx-y_minHx))
ErrorHy = sum((curl_y[85, :-1]-H_theta_y)**2)/(H_theta_y.shape[0]*(y_maxHy-y_minHy))

######################
# Plotting at y=0.25
######################
#Hx at y=0.25
plt.figure(figsize=(10, 6))  # Adjust figure size
plt.plot(X[85, :], curl_x[85, :], label="Hx - PINNs, for y=0.25", linewidth=2, linestyle='--', marker='o')
plt.plot(x_Hx, H_theta_x, label="Hx - MATLAB, for y=0.25", linewidth=2, linestyle='-', marker='s')

# Add labels, title, and legend
plt.xlabel("x-coordinate", fontsize=14)
plt.ylabel("Hx Component", fontsize=14)
plt.title("Comparison of Hx Components at x = 0.25", fontsize=16)
plt.legend(fontsize=12, loc='best')

# Customize ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()  # Adjust layout to fit labels and titles
plt.show()


#Hy at y=0.25
plt.figure(figsize=(10, 6))  # Adjust figure size
plt.plot(X[85, :], curl_y[85, :], label="Hy - PINNs, for y=0.25", linewidth=2, linestyle='--', marker='o')
plt.plot(x_Hy, H_theta_y, label="Hy - MATLAB, for y=0.25", linewidth=2, linestyle='-', marker='s')

# Add labels, title, and legend
plt.xlabel("x-coordinate", fontsize=14)
plt.ylabel("Hy Component", fontsize=14)
plt.title("Comparison of Hy Components at x = 0.25", fontsize=16)
plt.legend(fontsize=12, loc='best')

# Customize ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()  # Adjust layout to fit labels and titles
plt.show()



