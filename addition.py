

import matplotlib.pyplot as plt
import numpy as np

n_organisms = 100
n_problems = 100
n_inputs = 1 # per input_to_state
state_size = 2
n_iterations = 1000

def problem(xs):
    return np.sum(xs, axis=0)

def generate_data():
    x1 = np.random.normal(loc=0, scale=1, size=(1, n_problems, 1, n_inputs))
    x2 = np.random.normal(loc=0, scale=1, size=(1, n_problems, 1, n_inputs))
    xs = [x1, x2]
    y = problem(xs)
    return xs, y

def initialize_state(n_organisms, n_problems):
    return np.zeros((n_organisms, n_problems, 1, state_size))

def apply_nn(x, nn):
    for layer_index, layer in enumerate(nn):
        if layer_index != 0:
            x = np.maximum(0, x)

        x = np.matmul(x, layer)
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

def calculate_loss(read_nn, write_nn):
    xs, y = generate_data()
    y_predicted = predict(xs, read_nn, write_nn, n_organisms, n_problems)
    return loss(y, y_predicted)

def evolve(read_nn, write_nn):
    return (
        [np.random.normal(loc=x, scale=0.1) for x in read_nn],
        [np.random.normal(loc=x, scale=0.1) for x in write_nn]
    )

def main():

    # initialize organisms
    layer_size = 5
    read_nn = [
        np.zeros((n_organisms, 1, state_size + n_inputs, layer_size)),
        #np.zeros((n_organisms, 1, layer_size, layer_size)),
        #np.zeros((n_organisms, 1, layer_size, layer_size)),
        np.zeros((n_organisms, 1, layer_size, state_size))
    ]
    write_nn = [
        np.zeros((n_organisms, 1, state_size, layer_size)),
        #np.zeros((n_organisms, 1, layer_size, layer_size)),
        #np.zeros((n_organisms, 1, layer_size, layer_size)),
        np.zeros((n_organisms, 1, layer_size, 1))
    ]
    read_nn, write_nn = evolve(read_nn, write_nn)

    iteration_loss = []

    for iteration in range(n_iterations):

        loss_value = calculate_loss(read_nn, write_nn)
        iteration_loss.append(loss_value.min())

        print('iteration: {}, best loss value: {}'.format(iteration, loss_value.min()))

        organism_rank = np.argsort(loss_value)
        n_survivals = int(n_organisms/2)

        read_nn = [layer[organism_rank][:n_survivals] for layer in read_nn]
        write_nn = [layer[organism_rank][:n_survivals] for layer in write_nn]

        child_read_nn, child_write_nn = evolve(read_nn, write_nn)

        read_nn = [np.concatenate((layer, child_read_nn[layer_index])) for layer_index, layer in enumerate(read_nn)]
        write_nn = [np.concatenate((layer, child_write_nn[layer_index])) for layer_index, layer in enumerate(write_nn)]

    n_best_organisms = 50
    loss_value = calculate_loss(read_nn, write_nn)
    sorted_index = loss_value.argsort()
    best_read_nn = [layer[sorted_index[:n_best_organisms]] for layer in read_nn]
    best_write_nn = [layer[sorted_index[:n_best_organisms]] for layer in write_nn]
    print('final loss value: {}'.format(loss_value[sorted_index[0]]))

    plt.subplot(2, 2, 1)
    plt.plot(iteration_loss)
    plt.yscale('log')

    plt.subplot(2, 2, 2)
    n_points = 1000
    x1 = 1*np.ones((1, n_points, 1, n_inputs))
    x2 = np.ones((1, n_points, 1, n_inputs))
    x2[0, :, 0, 0] = np.linspace(start=-100, stop=100, num=n_points)
    xs = [x1, x2]
    y = problem(xs)
    plt.plot(x2[0,:,0,0], y[0,:,0,0])
    y_predicted = predict(xs, best_read_nn, best_write_nn, n_organisms=n_best_organisms, n_problems=n_points)
    plt.plot(x2[0,:,0,0], y_predicted[:,:,0,0].mean(axis=0))

    plt.subplot(2, 2, 3)
    plt.plot(x2[0,:,0,0], (y - y_predicted)[:,:,0,0].mean(axis=0))    
    plt.show()

if __name__ == '__main__':
    main()
