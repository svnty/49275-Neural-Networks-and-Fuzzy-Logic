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

w = np.array([0.2, 0.2, 0.3, 0.3, 0.1, 0.1, 0.4])

learning_rate = 0.1

def calculate_new_weight(w, x, d, r):
    return w + learning_rate * (d - r) * x

def TLU(v):
    if v >= 0:
        return 1
    else:
        return -1

# Part (i): Training for 12 cycles
print("Initial weights:", w)
print()

cycle_errors = []

for cycle in range(12):
  cycle_error = 0
  for index in range(1, 11):

    v = 0
    if index == 1:
        v = np.dot(w, x1)
    elif index == 2:
        v = np.dot(w, x2)
    elif index == 3:
        v = np.dot(w, x3)
    elif index == 4:
        v = np.dot(w, x4)
    elif index == 5:
        v = np.dot(w, x5)
    elif index == 6:
        v = np.dot(w, x6)
    elif index == 7:
        v = np.dot(w, x7)
    elif index == 8:
        v = np.dot(w, x8)
    elif index == 9:
        v = np.dot(w, x9)
    elif index == 10:
        v = np.dot(w, x10)

    r = TLU(v)
    d_val = 0
    if index == 1:
        d_val = d1[0]
    elif index == 2:
        d_val = d2[0]
    elif index == 3:
        d_val = d3[0]
    elif index == 4:
        d_val = d4[0]
    elif index == 5:
        d_val = d5[0]
    elif index == 6:
        d_val = d6[0]
    elif index == 7:
        d_val = d7[0]
    elif index == 8:
        d_val = d8[0]
    elif index == 9:
        d_val = d9[0]
    elif index == 10:
        d_val = d10[0]
    
    cycle_error += 0.5 * (d_val - r) ** 2
    print(f"Step {cycle*10+index}: x{index}, v = {v:.4f}, r = {r}, d = {d_val}")
    
    if index == 1:
        w = calculate_new_weight(w, x1, d1, r)
    elif index == 2:
        w = calculate_new_weight(w, x2, d2, r)
    elif index == 3:
        w = calculate_new_weight(w, x3, d3, r)
    elif index == 4:
        w = calculate_new_weight(w, x4, d4, r)
    elif index == 5:
        w = calculate_new_weight(w, x5, d5, r)
    elif index == 6:
        w = calculate_new_weight(w, x6, d6, r)
    elif index == 7:
        w = calculate_new_weight(w, x7, d7, r)
    elif index == 8:
        w = calculate_new_weight(w, x8, d8, r)
    elif index == 9:
        w = calculate_new_weight(w, x9, d9, r)
    elif index == 10:
        w = calculate_new_weight(w, x10, d10, r)

  cycle_errors.append(cycle_error)

print()
print("Final weight vector after 12 cycles:")
print(w)

# Part (ii): Verify final weights classify all samples
print()
print("Part (ii): Verification")

samples = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
desired = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10]

for i in range(10):
    v = np.dot(w, samples[i])
    r = TLU(v)
    d = desired[i][0]

    status = "Null"
    if r == d:
        status = "Correct"
    else:
        status = "Wrong"
    
    print(f"x{i+1}: v = {v:.4f}, TLU = {r}, desired = {d}, {status}")

# part (iii): error curve for 12 cycles
print()
for i in range(12):
    print(f"Cycle {i+1}: Ec = {cycle_errors[i]}")
print()

# part (iv): new learning rate
w = np.array([0.2, 0.2, 0.3, 0.3, 0.1, 0.1, 0.4])
learning_rate = 0.5
new_cycle_errors = []

for cycle in range(12):
    cycle_error = 0
    for index in range(1, 11):
        v = 0
        if index == 1:
            v = np.dot(w, x1)
        elif index == 2:
            v = np.dot(w, x2)
        elif index == 3:
            v = np.dot(w, x3)
        elif index == 4:
            v = np.dot(w, x4)
        elif index == 5:
            v = np.dot(w, x5)
        elif index == 6:
            v = np.dot(w, x6)
        elif index == 7:
            v = np.dot(w, x7)
        elif index == 8:
            v = np.dot(w, x8)
        elif index == 9:
            v = np.dot(w, x9)
        elif index == 10:
            v = np.dot(w, x10)

        r = TLU(v)
        d_val = 0
        if index == 1:
            d_val = d1[0]
        elif index == 2:
            d_val = d2[0]
        elif index == 3:
            d_val = d3[0]
        elif index == 4:
            d_val = d4[0]
        elif index == 5:
            d_val = d5[0]
        elif index == 6:
            d_val = d6[0]
        elif index == 7:
            d_val = d7[0]
        elif index == 8:
            d_val = d8[0]
        elif index == 9:
            d_val = d9[0]
        elif index == 10:
            d_val = d10[0]
        
        cycle_error += 0.5 * (d_val - r) ** 2
        print(f"Step {cycle*10+index}: x{index}, v = {v:.4f}, r = {r}, d = {d_val}")
        
        if index == 1:
            w = calculate_new_weight(w, x1, d1, r)
        elif index == 2:
            w = calculate_new_weight(w, x2, d2, r)
        elif index == 3:
            w = calculate_new_weight(w, x3, d3, r)
        elif index == 4:
            w = calculate_new_weight(w, x4, d4, r)
        elif index == 5:
            w = calculate_new_weight(w, x5, d5, r)
        elif index == 6:
            w = calculate_new_weight(w, x6, d6, r)
        elif index == 7:
            w = calculate_new_weight(w, x7, d7, r)
        elif index == 8:
            w = calculate_new_weight(w, x8, d8, r)
        elif index == 9:
            w = calculate_new_weight(w, x9, d9, r)
        elif index == 10:
            w = calculate_new_weight(w, x10, d10, r)

    new_cycle_errors.append(cycle_error)

print()
print("Final weight vector after 12 cycles with new learning rate:")
print(w)

print()
print("Error curve for 12 cycles with new learning rate:")
for i in range(12):
    print(f"New_cycle {i+1}: Ec = {new_cycle_errors[i]}")
print()