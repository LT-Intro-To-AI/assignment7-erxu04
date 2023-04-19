from neural import NeuralNet

xor_data = [
    ([0,0], [0]), 
    ([0,1], [1]),
    ([1,0], [1]),
    ([1,1], [0])
    ]

xorn = NeuralNet(2, 1, 1)
xorn.train(xor_data)

print(xorn.test_with_expected(xor_data))

voter_opinion = [
    ([0.9, 0.6, 0.8, 0.3, 0.1], [1]),
    ([0.8, 0.8, 0.4, 0.6, 0.4], [1]),
    ([0.7, 0.2, 0.4, 0.6, 0.3], [1]),
    ([0.5, 0.5, 0.8, 0.4, 0.8], [0]),
    ([0.3, 0.1, 0.6, 0.8, 0.8], [0]),
    ([0.6, 0.3, 0.4, 0.3, 0.6], [0])
    
]

von = NeuralNet(5, 6, 1)
von.train(voter_opinion)

print(von.test_with_expected(voter_opinion))

# Evaluate with the test data
print(f"case 1: {von.evaluate([1, 1, 1, 0.1, 1])}")
print(f"case 2: {von.evaluate([0.5, 0.2, 0.1, 0.7, 0.7])}")
print(f"case 3: {von.evaluate([0.8, 0.3, 0.3, 0.3, 0.8])}")
print(f"case 4: {von.evaluate([0.8, 0.3, 0.3, 0.8, 0.3])}")
print(f"case 4: {von.evaluate([0.9, 0.8, 0.8, 0.3, 0.6])}")

