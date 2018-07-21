
# Ideas

Input network -> Internal state -> Output network

Next step?
* More advanced evolver
* Rename organisms? solver, symbiotic/evolver
* Problem is too easy?
* Compare with static evolver (constant scale)
* Does batch norm help?
* Herds?


Test Results

    def problem(x1, x2):
        return x1*np.square(x2)

5 layers
size: 10

symbiotic.py: 0.026 (single scale)
symbiotic2.py: 0.043 (two scales)
symbiotic3.py: 0.24 to 3.38 shaky (nn type, layer index, layer type)

Drift of parameters towards zero?


Should evolve networks be detached from problem networks???? Let the
evolve networks reattach to other problem networks?
Only mutate evolve networks every X number of problem generations?
One evolver per X number of problem networks

Why does symbiotic2 work better than symbiotic4? Was stronger when n_organisms = 1. Why?

