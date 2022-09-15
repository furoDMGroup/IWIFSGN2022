import numpy as np


class IntervalValuedFuzzySet(object):
    def __init__(self, lower_bound, upper_bound, order='xu yager'):
        self.numpy_representation = np.array([lower_bound, upper_bound])
        self.order = order

    @staticmethod
    def from_numpy(numpy_representation, order):
        i = IntervalValuedFuzzySet(0, 0, order)
        i.numpy_representation = numpy_representation
        return i

    @staticmethod
    def xu_yager_less_than(a, b):
        return (a[0] + a[1] < b[0] + b[1]) or (a[1] + a[0] == b[0] + b[1] and a[1] - a[0] <= b[1] - b[0])

    @staticmethod
    def partial_order(a, b):
        return a[0] <= b[0] and a[1] <= b[1]

    @staticmethod
    def lex_order_1(a, b):
        return (a[0] < b[0]) or (a[0] == b[0] and a[1] <= b[1])

    @staticmethod
    def lex_order_2(a, b):
        return (a[1] < b[1]) or (a[1] == b[1] and a[0] <= b[0])

    def __lt__(self, other):
        if self.order == 'xu yager':
            return self.xu_yager_less_than(self.numpy_representation, other.numpy_representation)
        if self.order == 'lex1':
            return self.lex_order_1(self.numpy_representation, other.numpy_representation)
        if self.order == 'lex2':
            return self.lex_order_2(self.numpy_representation, other.numpy_representation)
        if self.order == 'partial':
            return self.partial_order(self.numpy_representation, other.numpy_representation)

    def __eq__(self, other):
        return self.numpy_representation == other.numpy_representation

    def __str__(self):
        return str(self.numpy_representation)

    def __repr__(self):
        return str(self.numpy_representation)

    def __truediv__(self, other):
        if type(other) == float:
            return self.numpy_representation / other


if __name__ == '__main__':
    print('demo')
    a = IntervalValuedFuzzySet(0.5, 0.8)
    b = IntervalValuedFuzzySet(0.2, 0.9)
    c = IntervalValuedFuzzySet(1.0, 1.0)
    sets = [a, b, c]
    print(sorted(sets))
