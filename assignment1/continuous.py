import numpy as np

x1 = np.array([0.5, 0.1, 0.6, -0.7, -0.5, 0.2, 1])
x2 = np.array([-0.4, -0.8, 0.3, 0.5, 0.2, 0.1, 1])
x3 = np.array([-0.7, 0.7, -0.8, 0.6, 0.4, 0.5, 1])
x4 = np.array([1.2, 0.8, -0.4, 0.5, 0.6, -0.3, 1])
x5 = np.array([0.6, 0.4, 0.6, 1.5, -0.2, -0.5, 1])
x6 = np.array([0.7, -0.2, 1.5, 0.9, -0.3, -0.6, 1])
x7 = np.array([1, 0.2, 0.6, 0.3, -0.3, 1.5, 1])
x8 = np.array([0.3, 0.4, 1, -1, 2.1, -0.9, 1])
x9 = np.array([0.5, 0.3, 0.1, 0.2, 0.5, 0.9, 1])
x10 = np.array([-0.3, 0.2, 0.4, 0.5, 1.1, 1.2, 1])

d1 = np.array([1])
d2 = np.array([-1])
d3 = np.array([1])
d4 = np.array([-1])
d5 = np.array([1])
d6 = np.array([-1])
d7 = np.array([1])
d8 = np.array([-1])
d9 = np.array([1])
d10 = np.array([-1])

samples = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
desired = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10]

originalWeight = np.array([0.2, 0.2, 0.3, 0.3, 0.1, 0.1, 0.4])
w = originalWeight

learning_rate = 0.1

def bipolar_activation(v):
    return (1 - np.exp(-v)) / (1 + np.exp(-v))

def bipolar_derivative(v):
    return 0.5 * (1 - bipolar_activation(v)**2)

def update_weight(w, x, d, v):
    return w + learning_rate * (d - bipolar_activation(v)) * bipolar_derivative(v) * x

# Part (i): Training for 15 cycles
print("Initial weights:", w)
print()

cycle_errors = []
wCount = 1
w11 = None
w151 = None

for cycle in range(15):
    cycle_error = 0
    for index in range(10):
        v = np.dot(w, samples[index])
        r = bipolar_activation(v)
        d_val = desired[index][0]
        cycle_error += 0.5 * (d_val - r) ** 2
        step = cycle*10+index
        print(f"Step {step}: x{index}, v = {v:.4f}, r = {r:.4f}, d = {d_val}, c = {learning_rate}")
        w = update_weight(w, samples[index], desired[index], v)
        wCount += 1

        if wCount == 11:
            w11 = w
        elif wCount == 151:
            w151 = w

    cycle_errors.append(cycle_error)

# Part (ii): Error curve and classification with w11 and w151
print()
print("Part (ii): Error curve for 15 cycles")
for index in range(15):
    print(f"Cycle {index+1}: Ec = {cycle_errors[index]:.4f}")

# part (ii): classify with w11 and w151
print()
print("Classification with w11:")
for index in range(10):
    v = np.dot(w11, samples[index])
    r = bipolar_activation(v)
    d = desired[index][0]
    classified = None

    if d < 0 and r < 0:
        classified = True
    elif d > 0 and r > 0:
        classified = True
    else:
        classified = False
        all_correct = False

    print(f"x{index+1}: v = {v:.4f}, r = {r:.4f}, d = {d}, classified = {classified}")

print()
print("Classification with w151:")

for index in range(10):
    v = np.dot(w151, samples[index])
    r = bipolar_activation(v)
    d = desired[index][0]
    classified = None

    if d < 0 and r < 0:
        classified = True
    elif d > 0 and r > 0:
        classified = True
    else:
        classified = False

    print(f"x{index+1}: v = {v:.4f}, r = {r:.4f}, d = {d}, classified = {classified}")

# Part (iii): New learning rate = 0.5
print()
print("Part (iii): Training with learning rate = 0.5")

w = originalWeight
learning_rate = 0.5
new_cycle_errors = []

for cycle in range(15):
  cycle_error = 0

  for index in range(10):
    v = np.dot(w, samples[index])
    r = bipolar_activation(v)
    d_val = desired[index][0]
    cycle_error += 0.5 * (d_val - r) ** 2
    step = cycle*10+index
    print(f"Step {step}: x{index}, v = {v:.4f}, r = {r:.4f}, d = {d_val}, c = {learning_rate}")
    w = update_weight(w, samples[index], desired[index], v)

  new_cycle_errors.append(cycle_error)

print()
print("Error curve comparison:")
print(f"{'Cycle':<8} {'a=0.1':<12} {'a=0.5':<12}")
for index in range(15):
    print(f"{index+1:<8} {cycle_errors[index]:<12.4f} {new_cycle_errors[index]:<12.4f}")

# Part (iv): Solution for zero classification error: train for more cycles until error converges close to zero
print()
print("Part (iv): Training until classification error is zero")

w = originalWeight
learning_rate = 0.1
converged = False

for cycle in range(1000):
    for index in range(10):
        v = np.dot(w, samples[index])
        r = bipolar_activation(v)
        d_val = desired[index][0]
        w = update_weight(w, samples[index], desired[index], v)

    # Check if all samples are correctly classified
    all_correct = True
    for index in range(10):
        v = np.dot(w, samples[index])
        r = bipolar_activation(v)
        d = desired[index][0]

        if d < 0 and r < 0:
            classified = True
        elif d > 0 and r > 0:
            classified = True
        else:
            classified = False
            all_correct = False

    if all_correct:
        print(f"Zero classification error reached at cycle {cycle+1}")
        print(f"Weight vector: {w}")
        converged = True
        break

if not converged:
    print("Zero classification error was not reached within 1000 cycles")