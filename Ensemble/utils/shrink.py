import numpy as np

def l21shrink(epsilon, x):

    output = x.copy()
    norm = np.linalg.norm(x, ord=2, axis=0)
    for i in range(x.shape[1]):
        if norm[i] > epsilon:
            for j in range(x.shape[0]):
                output[j,i] = x[j,i] - epsilon * x[j,i] / norm[i]
        else:
            output[:,i] = 0.
    return output


def l1shrink(epsilon,x):

    output = np.copy(x)
    above_index = np.where(output > epsilon)
    below_index = np.where(output < -epsilon)
    between_index = np.where((output <= epsilon) & (output >= -epsilon))
    
    output[above_index[0], above_index[1]] -= epsilon
    output[below_index[0], below_index[1]] += epsilon
    output[between_index[0], between_index[1]] = 0
    return output