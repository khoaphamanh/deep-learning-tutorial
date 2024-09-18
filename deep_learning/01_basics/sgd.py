import numpy as np
import matplotlib.pyplot as plt

# parameters
learning_rate = 0.1
epoch = 101
w_init = 8


# loss function L = f(w) = w^2
def f(w):
    return w**2 + 2 * w - 4


# gradient of L
def df(w):
    return 2 * w + 2


# Stochastic Gradient Descent
def sgd(w_t, learning_rate):
    return w_t - learning_rate * df(w_t)


# Initialize variables
w_t = w_init
losses = []
weights = []
iterations = []

# Perform Stochastic Gradient Descent
for iter in range(epoch):
    # Calculate the current loss
    current_loss = f(w_t)

    # Save current values for visualization
    losses.append(current_loss)
    weights.append(w_t)
    iterations.append(iter)

    # Update w_t using SGD
    w_t = sgd(w_t=w_t, learning_rate=learning_rate)

    # Print every 10 iterations
    if iter % 10 == 0:
        print(f"Iteration {iter}:")
        print(f"  Current loss: {current_loss}")
        print(f"  Current w_t: {w_t}")
        print()

# Prepare values for plotting the loss function
w_values = np.linspace(-10, 10, 400)
loss_values = f(w_values)

# Visualization using Matplotlib
plt.figure(figsize=(8, 6))

# Plot the loss function
plt.plot(w_values, loss_values, label="$y = w^2$", color="blue")

# Plot the trajectory of weights with red arrows
for i in range(len(weights) - 1):
    w_current = weights[i]
    w_next = weights[i + 1]

    # Plot the point at w_current
    plt.plot(w_current, f(w_current), "o", color="red")

    # Add an arrow showing the direction of weight updates (in red)
    plt.arrow(
        w_current,
        f(w_current),
        w_next - w_current,
        f(w_next) - f(w_current),
        head_width=0.2,
        head_length=0.2,
        fc="red",
        ec="red",
    )

# Plot the final point at the last weight, but keep it the same color
plt.plot(weights[-1], f(weights[-1]), "o", color="red", markersize=6)

# Labels and grid
plt.title("SGD w updates on Loss Function $f(w) = w^2$")
plt.xlabel("w")
plt.ylabel("Loss $f(w)$")
plt.legend()
plt.grid(True)
plt.show()
