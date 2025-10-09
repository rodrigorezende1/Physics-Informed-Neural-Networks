import os

# Set the DDE_BACKEND environment variable to "tensorflow2"
os.environ['DDE_BACKEND'] = 'tensorflow'

import deepxde as dde
import numpy as np
import tensorflow as tf # DeepXLE uses TensorFlow backend
import matplotlib.pyplot as plt

dde.config.set_default_float("float64")

# --- 1. Define Constants and Problem Parameters ---
# Geometry
X_MIN, X_MAX = -1.0, 1.0  # meters
Y_MIN, Y_MAX = -0.6, 0.6  # meters

# Material properties (assume free space for simplicity)
mu0 = 4 * np.pi * 1e-7 # Permeability of free space (H/m)
eps0 = 8.854e-12     # Permittivity of free space (F/m)

# Wave parameters (example: 1 GHz frequency)
frequency = 50.5*10**6 # Hz
omega = 2 * np.pi * frequency
k_wavenumber = omega * np.sqrt(mu0 * eps0) # Wavenumber in vacuum

# Source parameters (cable)
source_x_center = 0.0  # Center of the cable (e.g., at the origin)
source_y_center = 0.0
cable_radius = 2 * 1e-2 # 20 mm converted to meters
current_total = 1.0  # Amps
Factor = 1#10**3

# --- 2. Define the Geometry ---
geom = dde.geometry.Rectangle([X_MIN, Y_MIN], [X_MAX, Y_MAX])

# --- 3. Define the Source Term (RHS of PDE) ---
# A 2D Gaussian approximation for the current density Jz
def source_term_Jz(x, y):
    # Standard deviation derived from radius (e.g., 3-sigma rule)
    sigma = cable_radius / 2.0 # Adjust this for desired spread
    
    # Gaussian magnitude for current density (J = I / Area)
    # Factor to make integral approximately current_total
    # For a 2D Gaussian, Integral(exp(-(x^2+y^2)/(2*sigma^2))) dx dy = 2*pi*sigma^2
    # So, J0 = current_total / (2 * np.pi * sigma**2)
    
    # However, for PINNs, we often just provide a spatial distribution and let NN learn scaling
    # Or, if exact integral is crucial, compute it.
    # For now, let's use a scaled Gaussian.
    
    r_sq = (x - source_x_center)**2 + (y - source_y_center)**2
    # Use tf.exp for TensorFlow compatibility
    return current_total / (np.pi * cable_radius**2) * tf.exp(-r_sq / (2 * sigma**2))

# The actual RHS term in the PDE is -mu * Jz
def rhs_Jz(inputs, outputs, X):
    x = inputs[:, 0:1]
    y = inputs[:, 1:2]
    # Assuming Jz is purely real, so Jz_imag = 0
    return -mu0 * source_term_Jz(x, y) # This will be added to the real part of PDE loss

# --- 4. Define the PDE (Helmholtz Equation) ---
def pde(inputs, outputs):
    # inputs are [x, y] coordinates
    # outputs are [A_real, A_imag] from the neural network
    
    A_real = outputs[:, 0:1]
    A_imag = outputs[:, 1:2]

    # Calculate second derivatives (Laplacian)
    dA_real_xx = dde.grad.hessian(A_real, inputs, i=0, j=0)
    dA_real_yy = dde.grad.hessian(A_real, inputs, i=1, j=1)
    
    dA_imag_xx = dde.grad.hessian(A_imag, inputs, i=0, j=0)
    dA_imag_yy = dde.grad.hessian(A_imag, inputs, i=1, j=1)

    # Calculate the source term for the current point (from x,y)
    # The PDE for real part is: d2A_R/dx2 + d2A_R/dy2 + k^2 * A_R + mu * J_z = 0
    # The PDE for imaginary part is: d2A_I/dx2 + d2A_I/dy2 + k^2 * A_I = 0
    
    # Note: RHS is already incorporated as -mu*Jz in the formula below.
    pde_real_residual = (dA_real_xx + dA_real_yy) + k_wavenumber**2 * A_real + Factor*rhs_Jz(inputs, outputs, inputs)
    pde_imag_residual = (dA_imag_xx + dA_imag_yy) + k_wavenumber**2 * A_imag
    
    return [pde_real_residual, pde_imag_residual]

# --- 5. Define Boundary Conditions ---

# --- 5.1. Soft-Constrained PEC BCs (y = -0.6, y = 0.6) ---
# A_real = 0 and A_imag = 0 on these boundaries

def on_boundary_y_min(x, on_boundary):
    return on_boundary and np.isclose(x[1], Y_MIN)

def on_boundary_y_max(x, on_boundary):
    return on_boundary and np.isclose(x[1], Y_MAX)

# Dirichlet BCs for real and imaginary parts
bc_pec_y_min_real = dde.icbc.DirichletBC(geom, lambda x: 0, on_boundary_y_min, component=0) # component=0 for A_real
bc_pec_y_min_imag = dde.icbc.DirichletBC(geom, lambda x: 0, on_boundary_y_min, component=1) # component=1 for A_imag

bc_pec_y_max_real = dde.icbc.DirichletBC(geom, lambda x: 0, on_boundary_y_max, component=0)
bc_pec_y_max_imag = dde.icbc.DirichletBC(geom, lambda x: 0, on_boundary_y_max, component=1)


# --- 5.2. Soft-Constrained Sommerfeld ABCs (x = -1.0, x = 1.0) ---

# Define boundary regions
def on_boundary_x_min(x, on_boundary):
    return on_boundary and np.isclose(x[0], X_MIN)

def on_boundary_x_max(x, on_boundary):
    return on_boundary and np.isclose(x[0], X_MAX)

# Sommerfeld condition functions for OperatorBC (outputs are A_real, A_imag from NN)
# At x = X_MIN (outgoing in -x): d(A_R)/dx + k*A_I = 0 AND d(A_I)/dx - k*A_R = 0
def sommerfeld_x_min_real(inputs, outputs, X):
    A_real = outputs[:, 0:1]
    A_imag = outputs[:, 1:2]
    dA_real_dx = dde.grad.jacobian(A_real, inputs, i=0, j=0) # d(A_real)/dx (component 0 w.r.t. input 0)
    return dA_real_dx + k_wavenumber * A_imag

def sommerfeld_x_min_imag(inputs, outputs, X):
    A_real = outputs[:, 0:1]
    A_imag = outputs[:, 1:2]
    dA_imag_dx = dde.grad.jacobian(A_imag, inputs, i=0, j=0) # d(A_imag)/dx (component 1 w.r.t. input 0)
    return dA_imag_dx - k_wavenumber * A_real

# At x = X_MAX (outgoing in +x): d(A_R)/dx - k*A_I = 0 AND d(A_I)/dx + k*A_R = 0
def sommerfeld_x_max_real(inputs, outputs, X):
    A_real = outputs[:, 0:1]
    A_imag = outputs[:, 1:2]
    dA_real_dx = dde.grad.jacobian(A_real, inputs, i=0, j=0)
    return dA_real_dx - k_wavenumber * A_imag

def sommerfeld_x_max_imag(inputs, outputs, X):
    A_real = outputs[:, 0:1]
    A_imag = outputs[:, 1:2]
    dA_imag_dx = dde.grad.jacobian(A_imag, inputs, i=0, j=0)
    return dA_imag_dx + k_wavenumber * A_real

# Define the list of all boundary conditions (all are soft now)
bcs = [
    bc_pec_y_min_real,
    bc_pec_y_min_imag,
    bc_pec_y_max_real,
    bc_pec_y_max_imag,
    dde.icbc.OperatorBC(geom, sommerfeld_x_min_real, on_boundary_x_min),
    dde.icbc.OperatorBC(geom, sommerfeld_x_min_imag, on_boundary_x_min),
    dde.icbc.OperatorBC(geom, sommerfeld_x_max_real, on_boundary_x_max),
    dde.icbc.OperatorBC(geom, sommerfeld_x_max_imag, on_boundary_x_max),
]

# --- 6. Define the Neural Network ---
# Input dimension: 2 (x, y)
# Output dimension: 2 (A_real, A_imag)
# NO output_transform here as all BCs are soft.
net = dde.nn.FNN([2] + [175] * 6 + [2], "tanh", "Glorot uniform")
# net.apply_output_transform(output_transform) # REMOVED

# --- 7. Create dde.data.PDE ---
data = dde.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=2000,
    num_boundary=400, # This now includes points on all four boundaries
    num_test=300,
    solution=None
)

# --- 8. Create and Compile the Model ---
model = dde.Model(data, net)

# Compile the model with an optimizer and loss weights
# Now we have:
# [PDE_real, PDE_imag,
#  BC_y_min_real, BC_y_min_imag, BC_y_max_real, BC_y_max_imag, (4 PEC BCs)
#  BC_x_min_real, BC_x_min_imag, BC_x_max_real, BC_x_max_imag] (4 Sommerfeld BCs)
# Total loss terms: 2 (PDE) + 4 (PEC) + 4 (Sommerfeld) = 10 terms.
# You will likely need to tune these weights. PEC BCs often need high weights.
# Sommerfeld ABCs, being derivative BCs, might also need higher weights than the PDE loss.
loss_weights = [1, 1,    # PDE_real, PDE_imag
                10, 10,  # PEC Y_MIN real, imag
                10, 10,  # PEC Y_MAX real, imag
                5, 5,    # Sommerfeld X_MIN real, imag
                5, 5]    # Sommerfeld X_MAX real, imag

# Compile the model
for i in range(0,4):#maybe range(1,3) is enough
    #model.compile("adam", lr=10**-(3+i), loss_weights=[1, 1, 1, 1, 1, 1])
    adamax_optimizer = tf.keras.optimizers.Adamax(learning_rate=10**-(3+i))
    #adamax_optimizer = tf.keras.optimizers.Nadam(learning_rate=10**-(3+i))
    
    model.compile(optimizer=adamax_optimizer, lr=10**-(3+i), loss_weights=loss_weights)

    # Train the model
    losshistory, train_state = model.train(iterations=20000)

# model.compile("L-BFGS", loss_weights=loss_weights)
# losshistory_lbfgs, train_state_lbfgs = model.train()


# # Compile the model
# for i in range(0,4):#maybe range(1,3) is enough
#     model.compile("adam", lr=10**-(3+i), loss_weights=loss_weights)
#     # Train the model
#     losshistory, train_state = model.train(iterations=15000)


# --- 9. Train the Model ---
#losshistory, train_state = model.train(epochs=10000)

# Optional: Switch to L-BFGS-B
# model.compile("L-BFGS-B", loss_weights=loss_weights) # Re-compile with L-BFGS-B
# losshistory_lbfgs, train_state_lbfgs = model.train()

# --- 10. Post-processing and Visualization (Optional) ---
# To evaluate the solution
# x_test = geom.random_points(10000)
# A_predicted = model.predict(x_test)
# A_real_pred = A_predicted[:, 0]
# A_imag_pred = A_predicted[:, 1]
# A_magnitude = np.sqrt(A_real_pred**2 + A_imag_pred**2)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)



# --- 1. Generate a structured grid for prediction ---
# Create meshgrid for prediction. Make sure dimensions match your actual domain.
# For a 200x120 grid (nx=200, ny=120) over [-1,1]x[-0.6,0.6]
nx_grid = 201 # Number of points in x (e.g., 200 divisions -> 201 points)
ny_grid = 121 # Number of points in y (e.g., 120 divisions -> 121 points)

x_coords = np.linspace(X_MIN, X_MAX, nx_grid)
y_coords = np.linspace(Y_MIN, Y_MAX, ny_grid)
X_grid, Y_grid = np.meshgrid(x_coords, y_coords)

# Flatten the grid for model prediction
xy_grid_flat = np.hstack((X_grid.flatten()[:, None], Y_grid.flatten()[:, None]))

# --- 2. Predict A_real and A_imag on the grid ---
# This assumes 'model' is your trained DeepXDE model
A_predicted = model.predict(xy_grid_flat)/(Factor)#
A_real_flat = A_predicted[:, 0]
A_imag_flat = A_predicted[:, 1]

# Reshape predicted A_real and A_imag into 2D arrays matching the grid shape
# np.meshgrid returns (ny, nx) shape by default
A_real_grid = A_real_flat.reshape(ny_grid, nx_grid)
A_imag_grid = A_imag_flat.reshape(ny_grid, nx_grid)

# --- 3. Compute grid spacings ---
# These are the step sizes for np.gradient
dx = x_coords[1] - x_coords[0]
dy = y_coords[1] - y_coords[0]

# --- 4. Compute partial derivatives of A_R and A_I ---
# np.gradient returns (gradient_along_rows, gradient_along_columns)
# So, for (ny, nx) array, it's (grad_y, grad_x)
dA_real_dy, dA_real_dx = np.gradient(A_real_grid, dy, dx)
dA_imag_dy, dA_imag_dx = np.gradient(A_imag_grid, dy, dx)

# --- 5. Compute complex derivatives of A_z ---
# d(Az)/dx = d(Ar)/dx + i * d(Ai)/dx
# d(Az)/dy = d(Ar)/dy + i * d(Ai)/dy
dAz_dx = dA_real_dx + 1j * dA_imag_dx
dAz_dy = dA_real_dy + 1j * dA_imag_dy


# --- 6. Compute complex H-field components ---
# H_x = (1/mu0) * d(Az)/dy
# H_y = -(1/mu0) * d(Az)/dx
Hx_complex = (1 / mu0) * dAz_dy
Hy_complex = (-1 / mu0) * dAz_dx

# --- 7. Compute the magnitude of the H-field vector ---
# Magnitude of a complex vector (Hx, Hy) is sqrt(|Hx|^2 + |Hy|^2)
# Remember |Hx|^2 = Hx_real^2 + Hx_imag^2
magnitude_H = np.sqrt(np.abs(Hx_complex)**2 + np.abs(Hy_complex)**2)

Hx =  np.abs(Hx_complex)
Hy = np.abs(Hy_complex)

# Downsample the grid by slicing
# step = 2  # Plot every 5th arrow
# X_downsampled = X[::step, ::step]
# Y_downsampled = Y[::step, ::step]
# curl_x_uniform = curl_x_uniform[::step, ::step]
# curl_y_uniform = curl_y_uniform[::step, ::step]
# magnitude_step = magnitude[::step, ::step]

#######
###H CST
#####
data = np.loadtxt("D:/PINNs/Data_H_field_Open_BC_v1.0.txt") ## import data

#Data
x_Hx = data[:,0]/1000
H_theta_x = data[:,1]

x_Hx = x_Hx[31:]
x_Hx = x_Hx[:45]
H_theta_x = H_theta_x[31:]
H_theta_x = H_theta_x[:45]

data2 = np.loadtxt("D:/PINNs/Data_Hy_field_Open_BC_v1.0.txt") ## import data

#H_theta = np.zeros([data[:,2].size,2])
x_Hy = data2[:,0]/1000
H_theta_y = data2[:,1] # do I need this?

######################
# Plotting at y=0.25
######################
#Hx at y=0.25
plt.figure(figsize=(10, 6))  # Adjust figure size
plt.plot(X_grid[85, :], Hx[85, :], label="Hx - PINNs, for y=0.25", linewidth=2, linestyle='--', marker='o')
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

#Hy at x=0.25
plt.figure(figsize=(10, 6))  # Adjust figure size
plt.plot(Y_grid[:,125], Hy[:,125], label="Hy - PINNs, for x=0.25", linewidth=2, linestyle='--', marker='o')
plt.plot(x_Hy, H_theta_y, label="Hy - MATLAB, for x=0.25", linewidth=2, linestyle='-', marker='s')

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




