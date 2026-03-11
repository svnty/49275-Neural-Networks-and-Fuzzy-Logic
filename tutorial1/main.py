import numpy as np

print("Hello World")

# weights and inputs
learning_rate = 2

x_1 = np.array([1, 1])
x_2 = np.array([-0.5, 1])
x_3 = np.array([3, 1])
x_4 = np.array([-2, 1])

d_1 = 1
d_2 = -1
d_3 = 1
d_4 = -1

w = np.array([-2.5, 1.75])

# functions
def activation_function(v):
    return ((1 - np.exp(-v))/(1 + np.exp(-v)))

def activation_derivative(v):
    return 1/2 * (1 - activation_function(v)**2)

def loss_function(d, r):
    return (1/2) * (d - r)**2

def loss_derivative(d, r):
    return (r - d)

def get_delta_w(x, d, r, v):
    return (learning_rate * (d - r) * activation_derivative(v) * x)

def update_weight(w, x, d, r, v):
    return (np.add(w, get_delta_w(x, d, r, v)))

# running config and output variables
number_of_times_w_updated = 0
pattern_error = []
cycle_error = []
x = 0
d = 0

while number_of_times_w_updated < 4000:
    for i in range(4):
        if i == 0:
            x = x_1
            d = d_1
        elif i == 1:
            x = x_2
            d = d_2
        elif i == 2:
            x = x_3
            d = d_3
        elif i == 3:
            x = x_4
            d = d_4
        v: float = np.dot(w, x)
        print("v=" + str(v))
        r: float = activation_function(v)
        print("r=" + str(r))
        pattern_error.append(loss_function(d,r))
        w = update_weight(w, x, d, r, v)
        number_of_times_w_updated = number_of_times_w_updated + 1
        print("w=" + str(w))

