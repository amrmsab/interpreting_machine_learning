"""
A tiny, self-contained influence functions demo on logistic regression.

This follows Koh & Liang (2017), "Understanding Black-box Predictions via
Influence Functions," ICML. The point is to show — in code that fits on a
screen — what the math is actually doing.

We train a logistic regression model on a toy 2D dataset, then ask:
    "For one specific test point, which training points were most responsible
     for the prediction the model just made?"

You can drop this into a Colab cell or run it locally. Only NumPy needed.
"""

import numpy as np

rng = np.random.default_rng(0)


# ----------------------------------------------------------------------
# 1. A toy binary classification dataset: two Gaussian blobs in 2D.
# ----------------------------------------------------------------------
n_per_class = 50
X0 = rng.normal(loc=[-2, 0], scale=1.0, size=(n_per_class, 2))
X1 = rng.normal(loc=[+2, 0], scale=1.0, size=(n_per_class, 2))
X = np.vstack([X0, X1])                          # shape (100, 2)
y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])  # labels 0/1

# Inject one "weird" training point — a class-1 example that has wandered
# deep into class-0 territory. We want influence functions to flag this.
X[0] = np.array([-3.0, 0.5])
y[0] = 1.0


# ----------------------------------------------------------------------
# 2. Train logistic regression with plain gradient descent.
#    Loss:    L(theta) = mean( -y log(p) - (1-y) log(1-p) )
#    where    p = sigmoid(X @ theta)
# ----------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

theta = np.zeros(X.shape[1])
lr = 0.1
for _ in range(2000):
    p = sigmoid(X @ theta)
    grad = X.T @ (p - y) / len(X)
    theta -= lr * grad


# ----------------------------------------------------------------------
# 3. The Hessian of the loss at theta.
#    For logistic regression this has a clean closed form:
#        H = (1/n) * sum_i  p_i (1 - p_i)  x_i x_i^T
#    The Hessian tells us how the loss curves around our fitted theta.
# ----------------------------------------------------------------------
p = sigmoid(X @ theta)
W = p * (1.0 - p)                        # diag of weights, shape (n,)
H = (X.T * W) @ X / len(X)               # (d, d)
H_inv = np.linalg.inv(H)


# ----------------------------------------------------------------------
# 4. Influence of a training point z_i on the loss at a test point z_test.
#
#    The closed-form result (Koh & Liang 2017, eq. 2):
#
#        I_up,loss(z_i, z_test) = - grad L(z_test)^T @ H^-1 @ grad L(z_i)
#
#    Read this in plain English:
#       "How much does the loss on z_test change if we slightly upweight z_i?"
#    Negative score = upweighting z_i makes the test loss lower (z_i HELPED).
#    Positive score = upweighting z_i makes the test loss higher (z_i HURT).
# ----------------------------------------------------------------------
def grad_loss(x, y_true, theta):
    """Gradient of the per-example logistic loss w.r.t. theta."""
    return (sigmoid(x @ theta) - y_true) * x


def influence(x_train, y_train, x_test, y_test, theta, H_inv):
    g_test = grad_loss(x_test, y_test, theta)
    g_train = grad_loss(x_train, y_train, theta)
    return -g_test @ H_inv @ g_train


# ----------------------------------------------------------------------
# 5. Pick a test point and rank every training point by influence.
# ----------------------------------------------------------------------
x_test = np.array([1.5, 0.0])            # clearly in class-1 territory
y_test = 1.0

scores = np.array([
    influence(X[i], y[i], x_test, y_test, theta, H_inv)
    for i in range(len(X))
])

# Most "helpful" points: most negative scores (drove loss down on z_test).
# Most "harmful" points: most positive scores (drove loss up on z_test).
helpful = np.argsort(scores)[:5]
harmful = np.argsort(scores)[-5:][::-1]

print(f"Test point: {x_test}, true label: {int(y_test)}")
print(f"Model's predicted prob of class 1: {sigmoid(x_test @ theta):.3f}\n")

print("Top 5 most HELPFUL training points (drove the prediction):")
for i in helpful:
    print(f"  idx={i:3d}  label={int(y[i])}  x={X[i].round(2)}  score={scores[i]:+.4f}")

print("\nTop 5 most HARMFUL training points (worked against the prediction):")
for i in harmful:
    print(f"  idx={i:3d}  label={int(y[i])}  x={X[i].round(2)}  score={scores[i]:+.4f}")

# Note that idx=0, our deliberately mislabeled / weird point, should show up
# near the top of the harmful list. That's the whole pitch: influence
# functions surface bad data without any retraining.
