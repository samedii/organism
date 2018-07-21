# evolv evolution

# symbiotic relationship between
# 1. problem solver (problem_nn)
# 2. evolver (evolver_nn)

# Problem solver: classical neural network
# Evolver: evolves itself and problem network

# Evolver input:
# Flat problem_nn + flat evolver_nn
# All zero when evolving the other


# First: add intercepts [check]
# Second: deep multiplication

# Sheep herder
# Fixed size networks. Flatten and read static size?

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from functools import reduce

# nn specification
# list of layers

# layer specification
# tuple of layer_weight, layer_bias

# layer_weight specification
# (n_organisms, n_problems, n_rows, n_cols)

# layer_bias specification
# (n_organisms, n_problems, 1, n_cols)


def flatten_organisms(organisms):
    size = (organisms[0][0].shape[0], -1)
    flat_layers = [[layer_weight.reshape(size), layer_bias.reshape(size)] for layer_weight, layer_bias in organisms]
    all_layers = reduce(lambda x,y: x+y, flat_layers)
    return np.concatenate(all_layers, axis=1)

def reshape_organisms(flat_organisms, size):
    nn = []
    start = 0
    for layer_weight_size, layer_bias_size in size:
        stop = start + np.prod(layer_weight_size)
        layer_weight = flat_organisms[:, start:stop].reshape([-1]+layer_weight_size)
        start = stop
        stop = start + np.prod(layer_bias_size)
        layer_bias = flat_organisms[:, start:stop].reshape([-1]+layer_bias_size)
        start = stop

        nn.append((layer_weight, layer_bias))
    return nn

def size_from_organisms(organisms):
    size = []
    for layer_weight, layer_bias in organisms:
        layer_weight_size = []
        for index, s in enumerate(layer_weight.shape):
            if index == 0:
                continue
            layer_weight_size.append(s)

        layer_bias_size = []
        for index, s in enumerate(layer_bias.shape):
            if index == 0:
                continue
            layer_bias_size.append(s)
        size.append((layer_weight_size, layer_bias_size))
    return size

def softplus(x, limit=30):
    x = x.copy()
    less_than = (x <= limit)
    x[less_than] = np.log(1.0 + np.exp(x[less_than]))
    return x

def apply_organisms(x, organisms):
    for layer_index, (layer_weight, layer_bias) in enumerate(organisms):
        if layer_index != 0:
            x = np.maximum(0, x)

        #normalize_axis = (1, 2, 3)
        #x = (x - x.mean(axis=normalize_axis, keepdims=True))/(1e-4 + x.std(axis=normalize_axis, keepdims=True))
        x = np.matmul(x, layer_weight)
        x += layer_bias
    return x

def create_evolve_data(organisms, is_evolve):
    n_organisms = organisms[0][0].shape[0]
    is_evolve_factor = (1 if is_evolve else 0)

    data = []
    for layer_index, (layer_weight, layer_bias) in enumerate(organisms):
        layer_weight = layer_weight.reshape((n_organisms, -1, 1, 1))
        weight_factor = is_evolve_factor
        weight = weight_factor*np.ones_like(layer_weight)
        data.append(weight.reshape((n_organisms, -1, 1, 1)))

        layer_bias = layer_bias.reshape((n_organisms, -1, 1, 1))
        bias_factor = is_evolve_factor
        bias = bias_factor*np.ones_like(layer_bias)
        data.append(bias.reshape((n_organisms, -1, 1, 1)))

    return np.concatenate(data, axis=1)

def evolve(problem_organisms, evolve_organisms):
    n_organisms = problem_organisms[0][0].shape[0]
    decay_factor = 1.0#0.999
    minimum_scale = 0.0#1e-10


    evolve_data = create_evolve_data(evolve_organisms, is_evolve=True)
    evolve_scale = softplus(apply_organisms(evolve_data, evolve_organisms)) + minimum_scale
    evolve_scale = evolve_scale.reshape((n_organisms, -1))
    print(evolve_scale.mean())
    flat_evolve_organisms = flatten_organisms(evolve_organisms)
    child_evolve_organisms = np.random.normal(loc=flat_evolve_organisms, scale=evolve_scale, size=flat_evolve_organisms.shape)
    child_evolve_organisms = child_evolve_organisms*decay_factor
    child_evolve_organisms = reshape_organisms(child_evolve_organisms, size_from_organisms(evolve_organisms))


    evolve_data = create_evolve_data(problem_organisms, is_evolve=False)
    problem_scale = softplus(apply_organisms(evolve_data, child_evolve_organisms)) + minimum_scale
    problem_scale = problem_scale.reshape((n_organisms, -1))
    print(problem_scale.mean())
    flat_problem_organisms = flatten_organisms(problem_organisms)
    child_problem_organisms = np.random.normal(loc=flat_problem_organisms, scale=problem_scale, size=flat_problem_organisms.shape)
    child_problem_organisms = reshape_organisms(child_problem_organisms, size_from_organisms(problem_organisms))

    return child_problem_organisms, child_evolve_organisms

def init_layer(n_organisms, n_rows, n_cols):
    return (
        np.random.normal(loc=0, scale=0.01, size=(n_organisms, 1, n_rows, n_cols)),
        np.random.normal(loc=0, scale=0.01, size=(n_organisms, 1, 1, n_cols))
    )

def init_organisms(n_organisms, size):
    size = np.array(size)
    from_size = size[:-1]
    to_size = size[1:]
    return [init_layer(n_organisms, from_size[i], to_size[i]) for i in range(len(size)-1)]

def predict(X, problem_organisms):
    return apply_organisms(X, problem_organisms)

def loss(y, y_predicted):
    return np.square(y - y_predicted).mean(axis=(2, 3)).mean(axis=1)

def calculate_loss(X, y, problem_organisms):
    y_predicted = predict(X, problem_organisms)
    return loss(y, y_predicted)

#def problem(x1, x2):
#    return x1+x2

#def problem(x1, x2):
#    return x1*x2

def problem(x1, x2):
    return np.square(x1)*np.square(x2)

def generate_data(n_problems):
    # n_organisms, n_problems, 1 x n_inputs (row vector)
    x1 = np.random.normal(loc=0, scale=1, size=(1, n_problems, 1, 1))
    x2 = np.random.normal(loc=0, scale=1, size=(1, n_problems, 1, 1))
    X = np.concatenate([x1, x2], axis=3)
    y = problem(x1, x2)
    return X, y

def main():

    np.random.seed(100)

    n_organisms = 100
    n_problems = 500
    n_iterations = 1000
    problem_layer_size = 10

    n_best_organisms = 10

    X, y = generate_data(n_problems)

    # symbiotic
    problem_organism_size = (
        [X.shape[-1]] +
        [problem_layer_size for _ in range(3)] + 
        [y.shape[-1]]
    )
    evolve_organism_size = [1, 1]
    # Need to compress input. A RNN or just take in compressed input

    problem_organisms = init_organisms(n_organisms, problem_organism_size)
    evolve_organisms = init_organisms(n_organisms, evolve_organism_size)

    iteration_loss = []
    for iteration in range(n_iterations):

        #X, y = generate_data(n_problems)

        loss_value = calculate_loss(X, y, problem_organisms)
        iteration_loss.append(loss_value.min())

        print('iteration: {}, best loss value: {}'.format(iteration, loss_value.min()))

        organism_rank = np.argsort(loss_value)
        n_survivals = int(n_organisms/2)

        problem_organisms = [(layer_weight[organism_rank][:n_survivals], layer_bias[organism_rank][:n_survivals]) for layer_weight, layer_bias in problem_organisms]
        evolve_organisms = [(layer_weight[organism_rank][:n_survivals], layer_bias[organism_rank][:n_survivals]) for layer_weight, layer_bias in evolve_organisms]

        child_problem_organisms, child_evolve_organisms = evolve(problem_organisms, evolve_organisms)

        problem_organisms = [
            (
                np.concatenate((layer_weight, child_problem_organisms[layer_index][0])),
                np.concatenate((layer_bias, child_problem_organisms[layer_index][1]))
            )
            for layer_index, (layer_weight, layer_bias) in enumerate(problem_organisms)
        ]
        evolve_organisms = [
            (
                np.concatenate((layer_weight, child_evolve_organisms[layer_index][0])),
                np.concatenate((layer_bias, child_evolve_organisms[layer_index][1]))
            )
            for layer_index, (layer_weight, layer_bias) in enumerate(evolve_organisms)
        ]

    
    loss_value = calculate_loss(X, y, problem_organisms)
    sorted_index = loss_value.argsort()
    best_problem_organisms = [(layer_weight[sorted_index[:n_best_organisms]], layer_bias[sorted_index[:n_best_organisms]]) for layer_weight, layer_bias in problem_organisms]
    best_evolve_organisms = [(layer_weight[sorted_index[:n_best_organisms]], layer_bias[sorted_index[:n_best_organisms]]) for layer_weight, layer_bias in evolve_organisms]
    print('final loss value: {}'.format(loss_value[sorted_index[0]]))

    plt.subplot(2, 2, 1)
    plt.plot(iteration_loss)
    plt.yscale('log')

    plt.subplot(2, 2, 2)
    n_points = 1000
    x1 = 1*np.ones((1, n_points, 1, 1))
    x2 = np.ones((1, n_points, 1, 1))
    x2[0, :, 0, 0] = np.linspace(start=-5, stop=5, num=n_points)
    X = np.concatenate([x1, x2], axis=3)
    y = problem(x1, x2)
    plt.plot(x2[0,:,0,0], y[0,:,0,0])
    y_predicted = predict(X, best_problem_organisms)
    plt.plot(x2[0,:,0,0], y_predicted[:,:,0,0].mean(axis=0))

    plt.subplot(2, 2, 3)
    n_points = 1000
    x1 = 2*np.ones((1, n_points, 1, 1))
    x2 = np.ones((1, n_points, 1, 1))
    x2[0, :, 0, 0] = np.linspace(start=-5, stop=5, num=n_points)
    X = np.concatenate([x1, x2], axis=3)
    y = problem(x1, x2)
    plt.plot(x2[0,:,0,0], y[0,:,0,0])
    y_predicted = predict(X, best_problem_organisms)
    plt.plot(x2[0,:,0,0], y_predicted[:,:,0,0].mean(axis=0))

    plt.subplot(2, 2, 4)
    n_points = 1000
    x1 = -1*np.ones((1, n_points, 1, 1))
    x2 = np.ones((1, n_points, 1, 1))
    x2[0, :, 0, 0] = np.linspace(start=-5, stop=5, num=n_points)
    X = np.concatenate([x1, x2], axis=3)
    y = problem(x1, x2)
    plt.plot(x2[0,:,0,0], y[0,:,0,0])
    y_predicted = predict(X, best_problem_organisms)
    plt.plot(x2[0,:,0,0], y_predicted[:,:,0,0].mean(axis=0))

    plt.show()


if __name__ == '__main__':
    main()
