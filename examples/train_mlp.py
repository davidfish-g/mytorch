import numpy as np
from mytorch.nn import Linear, ReLU, CrossEntropyLoss
from mytorch.optim import SGD


class MLP:
    def __init__(self):
        self.layers = [Linear(2, 16), ReLU(), Linear(16, 2)]

    def forward(self, A):
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, dLdA):
        for layer in reversed(self.layers):
            dLdA = layer.backward(dLdA)


X = np.random.randn(256, 2)
Y = np.eye(2)[(X[:, 0] * X[:, 1] > 0).astype(int)]

model = MLP()
criterion = CrossEntropyLoss()
optimizer = SGD(model, lr=0.1, momentum=0.9)

for epoch in range(500):
    loss = criterion.forward(model.forward(X), Y)
    model.backward(criterion.backward())
    optimizer.step()
    if epoch % 100 == 0:
        print(f"epoch {epoch:3d}  loss {loss:.4f}")

accuracy = np.mean(model.forward(X).argmax(axis=1) == Y.argmax(axis=1))
print(f"training accuracy {accuracy:.0%}")
