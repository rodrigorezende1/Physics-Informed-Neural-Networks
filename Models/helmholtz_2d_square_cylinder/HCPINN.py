import os

# Set the DDE_BACKEND environment variable to "tensorflow2"
os.environ['DDE_BACKEND'] = 'tensorflow'

import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


dim1=1
dim2=1.2
freq=50.5*10**6
c0 = 299792458
mu0 = 4*np.pi*10**(-7)
eps0 = ((1/c0)**2)/mu0
omega = 2*np.pi*freq

k1 = (omega**2)*eps0*mu0
k2 = mu0
k3=1
hard_constraint = True
#hard_constraint = False
Factor = 1000

dde.config.set_default_float("float64")

# Define the PDE
def pde(x, f):
    """
    Helmholtz equation with Cylinder J = 1 source.
    """
    f_xx = dde.grad.hessian(f, x, i=0, j=0)  # Second derivative w.r.t. x
    f_yy = dde.grad.hessian(f, x, i=1, j=1)  # Second derivative w.r.t. y
    # Approximation for Cylinder J = 1
    delta = Factor/(np.pi*0.02**2)*(tf.cast(tf.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2) <= 0.02, tf.float64))
    return (-f_xx - f_yy - k1 * f - k2 * delta)*k3

#------------approx. Cylinder---------------------
# def current_density(x, y, radius=0.02):
#    r = np.sqrt(x**2 + y**2)  # Calculate the radial distance
#    return 1/(np.pi*radius**2)*(r <= radius).astype(float) 
#    #return 10**3/(np.pi*radius**2)*(r <= radius).astype(float) 
    
# x = np.linspace(-dim1/2, dim1/2, 101)
# y = np.linspace(-dim2/2, dim2/2, 121)
# xx, yy= np.meshgrid(x,y)
# zz=current_density(xx, yy)

# fig = plt.figure(figsize =(14, 9))
# ax = plt.axes(projection ='3d')
# ax.scatter(xx, yy, zz)
# plt.show()


# # Define the domain
# geom = dde.geometry.Rectangle([-dim1/2, -dim2/2], [dim1/2, dim2/2])

# Define the domain (Rectangular Waveguide)
geom = dde.geometry.Rectangle([-dim1/2, -dim2/2], [dim1/2, dim2/2])

# Define the boundary condition: f(x, y) = 0 on the boundary
def boundary(x, on_boundary):
    return on_boundary

if hard_constraint == True:
    bc = []
else:
    bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)
    #bc = dde.icbc.DirichletBC(geom, lambda x: -5.0e-05, boundary)

# Set the PDE problem
data = dde.data.PDE(geom, pde, bc, num_domain=1000, num_boundary=0, num_test=100)
#data = dde.data.PDE(geom, pde, bc, num_domain=1000, num_boundary=500)

# Define the neural network
#net = dde.maps.FNN([2] + [50] * 3 + [1], "sin", "Glorot uniform")
#net = dde.maps.FNN([2] + [120] * 6 + [1], "tanh", "Glorot uniform")
net = dde.maps.FNN([2] + [120] * 6 + [1], "tanh", "Glorot uniform")


def transform(x, y):
    res = (dim1/2 + x[:, 0:1]) * (x[:, 0:1]-dim1/2) * (dim2/2 + x[:, 1:2]) * (x[:, 1:2]-dim2/2) 
    return res * y


if hard_constraint == True:
    net.apply_output_transform(transform)

# Define the model
model = dde.Model(data, net)

# Compile the model
for i in range(1,4):#maybe range(1,3) is enough
    adamax_optimizer = tf.keras.optimizers.Adamax(lr=10**-3)
    model.compile(adamax_optimizer)

    # Train the model
    losshistory, train_state = model.train(iterations=15000)


dde.saveplot(losshistory, train_state, issave=False, isplot=True)

# Create a grid for visualization
X, Y = np.meshgrid(np.linspace(-dim1/2, dim1/2, 101), np.linspace(-dim2/2, dim2/2, 121))
xy = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

# Predict the solution
f_pred = model.predict(xy)/(Factor*1.4)
# Reshape for plotting
f_pred = f_pred.reshape(121, 101)

# Plot the contour
plt.figure(figsize=(8,  6))
plt.contourf(X, Y, f_pred, levels=50, cmap="viridis")
plt.colorbar(label="f(x, y)")
plt.title("PINNs Solution")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#Ploting BCs
plt.figure(figsize=(10, 6))  # Adjust figure size
plt.plot(Y[:, 0], f_pred[:, 0], label="Hx - PINNs, for y=0.25", linewidth=2, linestyle='--', marker='o')
plt.plot(Y[:, -1], f_pred[:, -1], label="Hx - PINNs, for y=0.25", linewidth=2, linestyle='--', marker='o')
plt.title("PINNs Solution at x=-0.5;0.5")
plt.show()

plt.figure(figsize=(10, 6))  # Adjust figure size
plt.plot(X[0, :], f_pred[0, :], label="Hx - PINNs, for y=0.25", linewidth=2, linestyle='--', marker='o')
plt.plot(X[-1, :], f_pred[-1, :], label="Hx - PINNs, for y=0.25", linewidth=2, linestyle='--', marker='o')
plt.title("PINNs Solution at y=-0.6;0.6")
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
data = np.loadtxt("C:/...") ## import data

#Data
x_Hx = data[:,0]
x_Hy = data[:,1]
#H_theta = np.zeros([data[:,2].size,2])
H_theta_x = data[:,2]
H_theta_y = data[:,3]


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








