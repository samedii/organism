
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

n_organisms = 100
n_problems = 100
n_inputs = 1 # per input_to_state
state_size = 6
n_iterations = 10000

n_read_layers = 2
n_write_layers = 3
read_layer_size = 10
write_layer_size = 10

def problem(xs):
    return xs[0]+xs[1]

#def problem(xs):
#    return xs[0]*xs[1]

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
    for layer_index, layer in enumerate(nn):
        if layer_index != 0:
            x = np.maximum(0, x)

        #normalize_axis = (3)
        #x = (x - x.mean(axis=normalize_axis, keepdims=True))/(1e-5 + x.std(axis=normalize_axis, keepdims=True))
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

def softplus(x, limit=30):
    x = x.copy()
    larger = (x <= limit)
    x[larger] = np.log(1.0 + np.exp(x[larger]))
    return x

def evolve(read_nn, write_nn):
    n_organisms = read_nn[0].shape[0]
    state = initialize_state(n_organisms, n_problems=1)

    one = np.ones((n_organisms, 1, 1, n_inputs))
    state = input_to_state(one*0, state, read_nn)
    state = input_to_state(one*0, state, read_nn)
    state = input_to_state(one*0, state, read_nn)

    child_read_nn = []
    for layer_index, layer in enumerate(read_nn):
        state = input_to_state(layer_index*one, state, read_nn)
        scale = softplus(state_to_output(state, write_nn))
        child_layer = np.random.normal(loc=layer, scale=scale, size=layer.shape)
        child_read_nn.append(child_layer)

    state = input_to_state(one, state, read_nn)
    state = input_to_state(one, state, read_nn)
    state = input_to_state(one, state, read_nn)

    child_write_nn = []
    for layer_index, layer in enumerate(write_nn):
        state = input_to_state(layer_index*one, state, read_nn)
        scale = softplus(state_to_output(state, write_nn))
        child_layer = np.random.normal(loc=layer, scale=scale, size=layer.shape)
        child_write_nn.append(child_layer)

    return (
        child_read_nn,
        child_write_nn
    )

def initialize_nns():
    read_nn = (
        [np.random.normal(size=(n_organisms, 1, state_size + n_inputs, read_layer_size))] +
        [np.random.normal(size=(n_organisms, 1, read_layer_size, read_layer_size)) for _ in range(n_read_layers - 2)] +
        [np.random.normal(size=(n_organisms, 1, read_layer_size, state_size))]
    )
    write_nn = (
        [np.random.normal(size=(n_organisms, 1, state_size, write_layer_size))] +
        [np.random.normal(size=(n_organisms, 1, write_layer_size, write_layer_size)) for _ in range(n_write_layers - 2)] +
        [np.random.normal(size=(n_organisms, 1, write_layer_size, 1))]
    )
    return read_nn, write_nn

def main():

    # initialize organisms
    read_nn, write_nn = initialize_nns()
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

    n_best_organisms = 10
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
    x2[0, :, 0, 0] = np.linspace(start=-3, stop=3, num=n_points)
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
