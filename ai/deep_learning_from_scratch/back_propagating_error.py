import numpy as np

"""
第5章 <误差反向传播法>
"""


# 乘法层
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    """
    正向传播
    """

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    """
    反向传播
    """

    def backwards(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


if __name__ == "__main__":
    apple = 100
    apple_num = 2
    tax = 1.1

    # layer
    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()

    # forward
    apple_price = mul_apple_layer.forward(apple, apple_num)
    price = mul_tax_layer.forward(apple_price, tax)
    print("price:", price)

    # backward
    dprice = 1
    dapple_price, dtax = mul_tax_layer.backwards(dprice)
    dapple, dapple_num = mul_apple_layer.backwards(dapple_price)
    print(dapple, dapple_num, dtax)

    # ReLU
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])
    relu = Relu()
    # print("ReLU forward:", relu.forward(x))
