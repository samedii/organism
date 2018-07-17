
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

n_organisms = 10
n_problems = 200
n_inputs = 1 # per input_to_state
state_size = 2
n_iterations = 3000
evolve_scale = 0.01

n_read_layers = 2
n_write_layers = 5
read_layer_size = 5
write_layer_size = 10

#def problem(xs):
#    return xs[0]+xs[1]

#def problem(xs):
#    return xs[0]*xs[1]

def problem(xs):
    return xs[0]*np.square(xs[1])

def generate_data():
    # n_organisms, n_problems, 1 x n_inputs (row vector)
    x1 = np.random.normal(loc=0, scale=1, size=(1, n_problems, 1, n_inputs))
    x2 = np.random.normal(loc=0, scale=1, size=(1, n_problems, 1, n_inputs))
    xs = np.stack([x1, x2])
    y = problem(xs)
    return xs, y

def initialize_state(n_organisms, n_problems):
    return np.zeros((n_organisms, n_problems, 1, state_size))

def apply_nn(x, nn):
    for layer_index, (layer_weight, layer_bias) in enumerate(nn):
        if layer_index != 0:
            x = np.maximum(0, x)

        normalize_axis = (1)
        x = (x - x.mean(axis=normalize_axis, keepdims=True))/(1e-4 + x.std(axis=normalize_axis, keepdims=True))
        x = np.matmul(x, layer_weight)
        x += layer_bias
    return x

def input_to_state(input, state, read_nn):
    x = np.concatenate((input, state), axis=3)
    return apply_nn(x, read_nn)

def state_to_output(state, write_nn):
    return apply_nn(state, write_nn)

def predict(xs, read_nn, write_nn, n_organisms, n_problems):
    state = initialize_state(n_organisms, n_problems)

    for x in xs:
        x = np.repeat(x, state.shape[0], axis=0)
        state = input_to_state(x, state, read_nn)

    return state_to_output(state, write_nn)

def loss(y, y_predicted):
    return np.square(y - y_predicted).mean(axis=(1, 2, 3))

def calculate_loss(xs, y, read_nn, write_nn):
    y_predicted = predict(xs, read_nn, write_nn, n_organisms, n_problems)
    return loss(y, y_predicted)

def softplus(x, limit=30):
    x = x.copy()
    larger = (x <= limit)
    x[larger] = np.log(1.0 + np.exp(x[larger]))
    return x

def evolve(read_nn, write_nn):
    return (
        [
            (
                np.random.normal(loc=layer_weight, scale=evolve_scale, size=layer_weight.shape),
                np.random.normal(loc=layer_bias, scale=evolve_scale, size=layer_bias.shape)
            ) for layer_weight, layer_bias in read_nn
        ],
        [
            (
                np.random.normal(loc=layer_weight, scale=evolve_scale, size=layer_weight.shape),
                np.random.normal(loc=layer_bias, scale=evolve_scale, size=layer_bias.shape)
            ) for layer_weight, layer_bias in write_nn
        ]
    )

def init_layer(n_organisms, n_rows, n_cols):
    return (
        np.random.normal(size=(n_organisms, 1, n_rows, n_cols)),
        np.random.normal(size=(n_organisms, 1, 1, n_cols))
    )

def init_nns():
    read_nn = (
        [init_layer(n_organisms, state_size + n_inputs, read_layer_size)] +
        [init_layer(n_organisms, read_layer_size, read_layer_size) for _ in range(n_read_layers - 2)] +
        [init_layer(n_organisms, read_layer_size, state_size)]
    )
    write_nn = (
        [init_layer(n_organisms, state_size, write_layer_size)] +
        [init_layer(n_organisms, write_layer_size, write_layer_size) for _ in range(n_write_layers - 2)] +
        [init_layer(n_organisms, write_layer_size, state_size)]
    )
    return read_nn, write_nn

def main():

    # initialize organisms
    read_nn, write_nn = init_nns()
    read_nn, write_nn = evolve(read_nn, write_nn)

    xs, y = generate_data()

    iteration_loss = []
    for iteration in range(n_iterations):

        loss_value = calculate_loss(xs, y, read_nn, write_nn)
        iteration_loss.append(loss_value.min())

        print('iteration: {}, best loss value: {}'.format(iteration, loss_value.min()))

        organism_rank = np.argsort(loss_value)
        n_survivals = int(n_organisms/2)

        read_nn = [(layer_weight[organism_rank][:n_survivals], layer_bias[organism_rank][:n_survivals]) for layer_weight, layer_bias in read_nn]
        write_nn = [(layer_weight[organism_rank][:n_survivals], layer_bias[organism_rank][:n_survivals]) for layer_weight, layer_bias in write_nn]

        child_read_nn, child_write_nn = evolve(read_nn, write_nn)

        read_nn = [
            (
                np.concatenate((layer_weight, child_read_nn[layer_index][0])),
                np.concatenate((layer_bias, child_read_nn[layer_index][1]))
            )
            for layer_index, (layer_weight, layer_bias) in enumerate(read_nn)
        ]
        write_nn = [
            (
                np.concatenate((layer_weight, child_write_nn[layer_index][0])),
                np.concatenate((layer_bias, child_write_nn[layer_index][1]))
            )
            for layer_index, (layer_weight, layer_bias) in enumerate(write_nn)
        ]

    n_best_organisms = 10
    loss_value = calculate_loss(xs, y, read_nn, write_nn)
    sorted_index = loss_value.argsort()
    best_read_nn = [(layer_weight[sorted_index[:n_best_organisms]], layer_bias[sorted_index[:n_best_organisms]]) for layer_weight, layer_bias in read_nn]
    best_write_nn = [(layer_weight[sorted_index[:n_best_organisms]], layer_bias[sorted_index[:n_best_organisms]]) for layer_weight, layer_bias in write_nn]
    print('final loss value: {}'.format(loss_value[sorted_index[0]]))

    plt.subplot(2, 2, 1)
    plt.plot(iteration_loss)
    plt.yscale('log')

    plt.subplot(2, 2, 2)
    n_points = 1000
    x1 = 1*np.ones((1, n_points, 1, n_inputs))
    x2 = np.ones((1, n_points, 1, n_inputs))
    x2[0, :, 0, 0] = np.linspace(start=-3, stop=3, num=n_points)
    xs = [x1, x2]
    y = problem(xs)
    plt.plot(x2[0,:,0,0], y[0,:,0,0])
    y_predicted = predict(xs, best_read_nn, best_write_nn, n_organisms=n_best_organisms, n_problems=n_points)
    plt.plot(x2[0,:,0,0], y_predicted[:,:,0,0].mean(axis=0))

    plt.subplot(2, 2, 3)
    plt.plot(x2[0,:,0,0], (y - y_predicted)[:,:,0,0].mean(axis=0))    

    plt.subplot(2, 2, 4)
    n_points = 1000
    x1 = 2*np.ones((1, n_points, 1, n_inputs))
    x2 = np.ones((1, n_points, 1, n_inputs))
    x2[0, :, 0, 0] = np.linspace(start=-3, stop=3, num=n_points)
    xs = [x1, x2]
    y = problem(xs)
    plt.plot(x2[0,:,0,0], y[0,:,0,0])
    y_predicted = predict(xs, best_read_nn, best_write_nn, n_organisms=n_best_organisms, n_problems=n_points)
    plt.plot(x2[0,:,0,0], y_predicted[:,:,0,0].mean(axis=0))

    plt.show()

if __name__ == '__main__':
    main()
