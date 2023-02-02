from abc import ABC, abstractmethod
import numpy as np
from scipy.stats.mstats import gmean


class Aggregation(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def aggregate_numpy_arrays_representation(self, fuzzy_sets):
        """
        :param fuzzy_sets: a numpy array holding fuzzy sets represented directly as numpy arrays
        :return: a fuzzy set, a numpy array result of aggregation
        """
        pass

    def aggregate_interval_valued_fuzzy_sets(self, fuzzy_sets):
        """
        :param fuzzy_sets: a numpy array holding fuzzy sets as IntervalValuedFuzzySet class instances
        :return: a fuzzy set, result of aggregation
        """
        fuzzy_sets_as_numpy = np.array([f.numpy_representation for f in fuzzy_sets])
        return self.aggregate_numpy_arrays_representation(fuzzy_sets_as_numpy)

    @staticmethod
    def change_aggregation_to_name(agg):
        if isinstance(agg, A1Aggregation):
            return 'A1'
        if isinstance(agg, A2Aggregation):
            return 'A2'
        if isinstance(agg, A3Aggregation):
            return 'A3'
        if isinstance(agg, A4Aggregation):
            return 'A4'
        if isinstance(agg, A5Aggregation):
            return 'A5'
        if isinstance(agg, A6Aggregation):
            return 'A6'
        if isinstance(agg, A7Aggregation):
            return 'A7'
        if isinstance(agg, A8Aggregation):
            return 'A8'
        if isinstance(agg, A9Aggregation):
            return 'A9'
        if isinstance(agg, A10Aggregation):
            return 'A10'
        if isinstance(agg, A11Aggregation):
            return 'A11'
        if isinstance(agg, A12Aggregation):
            return 'A12'
        if isinstance(agg, A13Aggregation):
            return 'A13'
        if isinstance(agg, A14Aggregation):
            return 'A14'
        if isinstance(agg, A15Aggregation):
            return 'A15'
        if isinstance(agg, A16Aggregation):
            return 'A16'
        if isinstance(agg, A17Aggregation):
            return 'A17'


# aggregations names comes from paper
class A1Aggregation(Aggregation):
    def __init__(self):
        super().__init__()

    def aggregate_numpy_arrays_representation(self, fuzzy_sets):
        return fuzzy_sets.sum(axis=0) / fuzzy_sets.shape[0]


class GMeanNumericallyImproved(Aggregation):
    def __init__(self, axis=0):
        super().__init__()
        self.axis = axis

    def aggregate_numpy_arrays_representation(self, fuzzy_sets):
        if len(fuzzy_sets.shape) > 1:
            raise RuntimeWarning('The implementation of this aggregation only allows'
                                 ' one dimensional array')
        if 0 in fuzzy_sets:
            return 0.0
        if 0.0 in fuzzy_sets:
            return 0.0
        return gmean(fuzzy_sets, axis=self.axis)


class A2Aggregation(Aggregation):
    def __init__(self):
        super().__init__()

    def _f(self, sum, upper, lower, n):
        sum -= upper
        sum += lower
        return sum / n

    def aggregate_numpy_arrays_representation(self, fuzzy_sets):
        summed = fuzzy_sets.sum(axis=0)
        t = np.array([self._f(summed[1], f[1], f[0], fuzzy_sets.shape[0]) for f in fuzzy_sets])
        # print(t)
        return np.array([summed[0] / fuzzy_sets.shape[0], np.max(t)])


class A3Aggregation(Aggregation):
    def __init__(self):
        super().__init__()

    def aggregate_numpy_arrays_representation(self, fuzzy_sets):
        summed = fuzzy_sets.sum(axis=0)
        # division by zero, here 0/0 = 0
        if summed[1] == 0:
            return np.array([summed[0] / fuzzy_sets.shape[0], 0])
        # standard way
        squared = np.square(fuzzy_sets[:, 1])
        return np.array([summed[0] / fuzzy_sets.shape[0], np.sum(squared, axis=0) / summed[1]])


class A4Aggregation(Aggregation):
    # parameter p should be integer >= 3
    def __init__(self, p=3):
        super().__init__()
        if p < 3:
            if p == 2:
                raise RuntimeWarning('Setting parameter p equal to 2 coincides to A3')
            raise RuntimeWarning('Parameter p should be integer >= 3')
        if p is float:
            raise RuntimeWarning('Parameter p should be integer')
        self.p = p

    def aggregate_numpy_arrays_representation(self, fuzzy_sets):
        summed = fuzzy_sets.sum(axis=0)
        # division by zero, here 0/0 = 0
        if summed[1] == 0:
            return np.array([summed[0] / fuzzy_sets.shape[0], 0])
        # standard way
        powered = np.power(fuzzy_sets[:, 1], self.p)
        powered_minus_one = np.power(fuzzy_sets[:, 1], self.p - 1)
        # print('powered', powered)
        return np.array([summed[0] / fuzzy_sets.shape[0], np.sum(powered, axis=0) / np.sum(powered_minus_one)])


class A5Aggregation(Aggregation):
    def __init__(self):
        super().__init__()

    def aggregate_numpy_arrays_representation(self, fuzzy_sets):
        lower = np.square(fuzzy_sets[:, 0])
        upper = np.float_power(fuzzy_sets[:, 1], 3)
        n = fuzzy_sets.shape[0]
        return np.array([np.sqrt(lower.sum(axis=0) / n), np.float_power(upper.sum(axis=0) / n, (1 / 3))])


class A6Aggregation(Aggregation):
    def __init__(self):
        super().__init__()

    def aggregate_numpy_arrays_representation(self, fuzzy_sets):
        lower = np.float_power(fuzzy_sets[:, 0], 3)
        upper = np.float_power(fuzzy_sets[:, 1], 4)
        n = fuzzy_sets.shape[0]
        return np.array(
            [np.float_power(lower.sum(axis=0) / n, (1 / 3)), np.float_power(upper.sum(axis=0) / n, (1 / 4))])


class A7Aggregation(Aggregation):
    def __init__(self):
        super().__init__()

    def _f(self, sum, upper, lower, n):
        sum -= lower
        sum += upper
        return sum / n

    def aggregate_numpy_arrays_representation(self, fuzzy_sets):
        summed = fuzzy_sets.sum(axis=0)
        t = np.array([self._f(summed[0], f[1], f[0], fuzzy_sets.shape[0]) for f in fuzzy_sets])
        return np.array([np.min(t), summed[1] / fuzzy_sets.shape[0]])


class A8Aggregation(Aggregation):
    def __init__(self):
        super().__init__()

    def aggregate_numpy_arrays_representation(self, fuzzy_sets):
        n = fuzzy_sets.shape[0]
        lower = GMeanNumericallyImproved(axis=0).aggregate_numpy_arrays_representation(fuzzy_sets[:, 0])
        upper_down = fuzzy_sets[:, 1].sum(axis=0)
        # division by zero, here 0/0 = 0
        if upper_down == 0:
            return np.array([lower, 0])
        upper_up = np.square(fuzzy_sets[:, 1]).sum(axis=0)
        return np.array([lower, upper_up / upper_down])


class A9Aggregation(Aggregation):
    def __init__(self):
        super().__init__()

    def aggregate_numpy_arrays_representation(self, fuzzy_sets):
        lower = np.square(fuzzy_sets[:, 0])
        n = fuzzy_sets.shape[0]
        upper_down = np.power(fuzzy_sets[:, 1], 2).sum(axis=0)
        # division by zero, here 0/0 = 0
        if upper_down == 0:
            return np.array([np.sqrt(lower.sum(axis=0) / n), 0])
        upper_up = np.power(fuzzy_sets[:, 1], 3).sum(axis=0)
        return np.array([np.sqrt(lower.sum(axis=0) / n), upper_up / upper_down])


class A10Aggregation(Aggregation):
    def __init__(self):
        super().__init__()

    def aggregate_numpy_arrays_representation(self, fuzzy_sets):
        lower = np.square(fuzzy_sets[:, 0])
        n = fuzzy_sets.shape[0]
        upper = np.square(fuzzy_sets[:, 1])
        return np.array([np.sqrt(lower.sum(axis=0) / n), np.sqrt(upper.sum(axis=0) / n)])


class A11Aggregation(Aggregation):
    def __init__(self, q, s):
        super().__init__()
        if s < q:
            raise RuntimeWarning('The parameter s could not be lower than q')
        if s == 0 or q == 0:
            raise RuntimeWarning('The parameters q and s can not be equal to 0')
        self.q = q
        self.s = s

    def aggregate_numpy_arrays_representation(self, fuzzy_sets):
        n = fuzzy_sets.shape[0]

        if self.s < 0 and 0 in fuzzy_sets[:, 1]:
            return np.array([0.0, 0.0])
        elif self.q < 0 and 0 in fuzzy_sets[:, 0]:
            return np.array(
                [0.0, np.float_power(np.float_power(fuzzy_sets[:, 1], self.s).sum(axis=0) / n, (1 / self.s))])
        else:
            lower = np.float_power(fuzzy_sets[:, 0], self.q)
            upper = np.float_power(fuzzy_sets[:, 1], self.s)
            return np.array([np.float_power((lower.sum(axis=0) / n), (1 / self.q)),
                             np.float_power(upper.sum(axis=0) / n, (1 / self.s))])


class A12Aggregation(Aggregation):
    def __init__(self, q):
        super().__init__()
        self.q = q
        if self.q == 0:
            raise RuntimeWarning('The parameter q can not be equal to 0')

    def _f(self, sum, upper, lower, n):
        sum -= upper
        sum += lower
        return np.float_power(sum / n, 1 / self.q)

    def aggregate_numpy_arrays_representation(self, fuzzy_sets):
        if self.q < 0 and 0 in fuzzy_sets[:, 1]:
            return np.array([0.0, 0.0])

        if self.q < 0:
            indexes = []
            mask = np.zeros(fuzzy_sets.shape, dtype=np.bool)
            for i in range(fuzzy_sets[:, 0].shape[0]):
                # print('i=', i)
                if fuzzy_sets[i, 0] == 0.0 and fuzzy_sets[i, 1] != 0.0:
                    indexes.append(i)
                    mask[i, 0] = True
            if len(indexes) == 0:
                powered = np.float_power(fuzzy_sets, self.q)
                summed = powered.sum(axis=0)
                t = np.array([self._f(summed[1], f[1], f[0], fuzzy_sets.shape[0]) for f in powered])
                return np.array(
                [np.float_power(summed[0] / fuzzy_sets.shape[0], (1 / self.q)), np.max(t)])

            # print(mask)
            # print(indexes)
            powered = np.ma.power(np.ma.MaskedArray(fuzzy_sets, mask=mask), self.q)
            summed = powered.sum(axis=0)
            ts = []
            for i in range(powered.shape[0]):
                if i not in indexes:
                    f = powered[i]
                    ts.append(self._f(summed[1], f[1], f[0], fuzzy_sets.shape[0]))
            # print(ts)
            if len(ts) == 0:
                return np.array([0.0, 0.0])
            t = np.array(ts)
            return np.array([0.0, np.max(t)])

        powered = np.float_power(fuzzy_sets, self.q)
        summed = powered.sum(axis=0)
        t = np.array([self._f(summed[1], f[1], f[0], fuzzy_sets.shape[0]) for f in powered])
        return np.array(
            [np.float_power(summed[0] / fuzzy_sets.shape[0], (1 / self.q)), np.max(t)])


class A13Aggregation(Aggregation):
    def __init__(self, q):
        super().__init__()
        if q == 0:
            raise RuntimeWarning('Parameter q could not be equal zero')
        self.q = q

    def _f(self, sum, upper, lower, n):
        sum -= upper
        sum += lower
        return np.log(sum / n)

    def aggregate_numpy_arrays_representation(self, fuzzy_sets):
        exp = np.exp(self.q * fuzzy_sets)
        summed = exp.sum(axis=0)
        t = np.array([self._f(summed[1], f[1], f[0], fuzzy_sets.shape[0]) for f in exp])
        return np.array([(1 / self.q) * np.log(summed[0] / fuzzy_sets.shape[0]), (1 / self.q) * np.max(t)])


class A14Aggregation(Aggregation):
    def __init__(self, q, s, p):
        super().__init__()
        if s == 0 or q == 0:
            raise RuntimeWarning('The parameters q and s can not be equal to 0')
        if p < 0 or p > 1:
            raise RuntimeWarning('p should be between 0 and 1')

        self.q = q
        self.s = s
        self.p = p

    def aggregate_numpy_arrays_representation(self, fuzzy_sets):
        # lower = np.float_power(fuzzy_sets[:, 0], self.q)
        # upper = np.float_power(fuzzy_sets[:, 1], self.s)
        n = fuzzy_sets.shape[0]

        if (self.q < 0 and 0 in fuzzy_sets[:, 0]) and self.s > 0:
            # if zero exists in lower boundaries, the minimum must be zero and the power mean with
            # q must be zero to avoid division by zero
            return np.array([0.0, self.p * (
                np.float_power((np.float_power(fuzzy_sets[:, 1], self.s).sum(axis=0) / n), (1 / self.s))) + (
                                     1 - self.p) * np.max(fuzzy_sets[:, 1])])

        if (self.q < 0 and 0 in fuzzy_sets[:, 0]) and (self.s < 0 and 0 in fuzzy_sets[:, 1]):
            return np.array([0.0, (1 - self.p) * np.max(fuzzy_sets[:, 1])])

        if (self.s < 0 and 0 in fuzzy_sets[:, 1]) and self.q > 0:
            return np.array([(1 - self.p) * (
                np.float_power((np.float_power(fuzzy_sets[:, 0], self.q).sum(axis=0) / n), (1 / self.q))),
                             (1 - self.p) * np.max(fuzzy_sets[:, 1])])

        if (self.s < 0 and 0 in fuzzy_sets[:, 1]) and self.q < 0:
            return np.array([0.0, (1 - self.p) * np.max(fuzzy_sets[:, 1])])
        if (self.q < 0 and 0 in fuzzy_sets[:, 0]) and (self.s < 0 and 0 not in fuzzy_sets[:, 1]):
            return np.array([0.0, self.p * (
                np.float_power((np.float_power(fuzzy_sets[:, 1], self.s).sum(axis=0) / n), (1 / self.s))) + (
                                     1 - self.p) * np.max(fuzzy_sets[:, 1])])

        lower = np.float_power(fuzzy_sets[:, 0], self.q)
        upper = np.float_power(fuzzy_sets[:, 1], self.s)
        return np.array([(self.p * np.min(fuzzy_sets[:, 0]) + (1 - self.p) * (
            np.float_power((lower.sum(axis=0) / n), (1 / self.q)))),
                         self.p * (np.float_power((upper.sum(axis=0) / n), (1 / self.s))) + (1 - self.p) * np.max(
                             fuzzy_sets[:, 1])])


class A15Aggregation(Aggregation):
    def __init__(self, p, q):
        super().__init__()
        self.p = p
        self.q = q
        if q <= 1:
            raise RuntimeWarning('q should be greater than 1')
        if p < 0 or p > 1:
            raise RuntimeWarning('p should be between 0 and 1')

    def aggregate_numpy_arrays_representation(self, fuzzy_sets):
        summed = fuzzy_sets.sum(axis=0)
        # division by zero, here 0/0 = 0
        if summed[1] == 0:
            return np.array([0.0, 0.0])
        # standard way
        powered_upper = np.float_power(fuzzy_sets[:, 1], self.q)
        squared = np.square(fuzzy_sets[:, 1])
        n = fuzzy_sets.shape[0]
        powered_minus_one = np.float_power(fuzzy_sets[:, 1], self.q - 1)
        return np.array([(1 - self.p) * (summed[0] / fuzzy_sets.shape[0]) + self.p *
                         GMeanNumericallyImproved(axis=0).aggregate_numpy_arrays_representation(fuzzy_sets[:, 0]),
                         self.p * (np.sum(squared, axis=0) / summed[1]) + (1 - self.p) * (
                                 np.sum(powered_upper, axis=0) / np.sum(powered_minus_one))])


class A16Aggregation(Aggregation):
    def __init__(self, q):
        super().__init__()
        self.q = q
        if self.q == 0:
            raise RuntimeWarning('The parameter q can not be equal to 0')

    def _f(self, sum, upper, lower, n):
        sum -= lower
        sum += upper
        return np.float_power(sum / n, 1 / self.q)

    def aggregate_numpy_arrays_representation(self, fuzzy_sets):
        if self.q < 0 and (0 in fuzzy_sets[:, 0] or 0 in fuzzy_sets[:, 1]):
            return np.array([0.0, 0.0])

        fuzzy_sets = np.float_power(fuzzy_sets, self.q)
        summed = fuzzy_sets.sum(axis=0)
        t = np.array([self._f(summed[0], f[1], f[0], fuzzy_sets.shape[0]) for f in fuzzy_sets])
        return np.array(
            [np.min(t), np.float_power(summed[1] / fuzzy_sets.shape[0], (1 / self.q))])


class A17Aggregation(Aggregation):
    def __init__(self, q):
        super().__init__()
        self.q = q

    def _f(self, sum, upper, lower, n):
        sum -= lower
        sum += upper
        return np.log(sum / n)

    def aggregate_numpy_arrays_representation(self, fuzzy_sets):
        exp = np.exp(self.q * fuzzy_sets)
        summed = exp.sum(axis=0)
        t = np.array([self._f(summed[0], f[1], f[0], fuzzy_sets.shape[0]) for f in exp])
        return np.array([(1 / self.q) * np.min(t), ((1 / self.q) * np.log(summed[1] / fuzzy_sets.shape[0]))])


if __name__ == '__main__':
    a = A1Aggregation().aggregate_numpy_arrays_representation(np.array([[0.1, 0.7], [0.3, 0.6]]))
    print(a)
    a = A4Aggregation(p=4).aggregate_numpy_arrays_representation(np.array([[0.1, 0.7], [0.3, 0.6]]))
    print(a)
    a = A5Aggregation().aggregate_numpy_arrays_representation(np.array([[0.1, 0.7], [0.3, 0.6]]))
    print(a)
    a = A6Aggregation().aggregate_numpy_arrays_representation(np.array([[0.1, 0.7], [0.3, 0.6]]))
    print(a)
    a = A7Aggregation().aggregate_numpy_arrays_representation(np.array([[0.1, 0.7], [0.3, 0.6]]))
    print(a)
    a = A8Aggregation().aggregate_numpy_arrays_representation(np.array([[0.1, 0.7], [0.3, 0.6]]))
    print(a)
    a = A9Aggregation().aggregate_numpy_arrays_representation(np.array([[0.1, 0.7], [0.3, 0.6]]))
    print(a)
    a = A10Aggregation().aggregate_numpy_arrays_representation(np.array([[0.1, 0.7], [0.3, 0.6]]))
    print(a)
    a = GMeanNumericallyImproved().aggregate_numpy_arrays_representation(np.array([0.0, 0.1]))
    print(a)
    print('A8 with div by zero')
    a = A8Aggregation().aggregate_numpy_arrays_representation(np.array([[0.0, 0.1], [0.3, 0.6]]))
    print(a)

    a = A1Aggregation().aggregate_numpy_arrays_representation(np.array([[0.1, 0.1], [0.9, 0.9], [0.6, 0.6]]))
    b = A2Aggregation().aggregate_numpy_arrays_representation(np.array([[0, 0.2], [0, 0.3], [0, 0.4]]))
    c = A3Aggregation().aggregate_numpy_arrays_representation(np.array([[0.1, 0.1], [0.9, 0.9], [0.6, 0.6]]))
    d = A4Aggregation().aggregate_numpy_arrays_representation(np.array([[0.1, 0.1], [0.9, 0.9], [0.6, 0.6]]))
    e = A5Aggregation().aggregate_numpy_arrays_representation(np.array([[0.1, 0.1], [0.9, 0.9], [0.6, 0.6]]))
    f = A6Aggregation().aggregate_numpy_arrays_representation(np.array([[0.5, 0.7], [0.3, 0.6], [0.2, 0.8]]))
    g = A7Aggregation().aggregate_numpy_arrays_representation(np.array([[0.5, 1], [0.3, 1], [0, 1]]))
    h = A8Aggregation().aggregate_numpy_arrays_representation(np.array([[0.1, 0.1], [0.9, 0.9], [0.6, 0.6]]))
    i = A9Aggregation().aggregate_numpy_arrays_representation(np.array([[0.1, 0.1], [0.9, 0.9], [0.6, 0.6]]))
    j = A10Aggregation().aggregate_numpy_arrays_representation(np.array([[0.1, 0.1], [0.9, 0.9], [0.6, 0.6]]))

    k = A11Aggregation(q=-2, s=1).aggregate_numpy_arrays_representation(np.array([[0.0, 0.5], [0.9, 0.9], [0.6, 0.6]]))
    l = A12Aggregation(q=-1).aggregate_numpy_arrays_representation(np.array([[0, 0.2], [0.3, 0.3], [0.0, 0.1], [0.2, 0.2]]))
    u = A13Aggregation(q=-0.5).aggregate_numpy_arrays_representation(np.array([[0.0, 0.5], [0.4, 0.9], [0.2, 0.6]]))
    x = A14Aggregation(q=1, s=-1, p=0.5).aggregate_numpy_arrays_representation(
        np.array([[0.0, 0.0], [0.9, 0.9], [0.6, 0.6]]))

    # y = A15Aggregation(p=0, q=3).aggregate_numpy_arrays_representation(np.array([[0.0, 0.1], [0.9, 0.9], [0.6, 0.6]]))

    # yy= A15Aggregation(p=0.5,r=3).aggregate_numpy_arrays_representation(np.array([[0, 0], [0, 0], [0, 0]]))
    w = A16Aggregation(q=-1.5).aggregate_numpy_arrays_representation(np.array([[0.5, 1], [0.3, 1], [0, 1]]))
    z = A17Aggregation(q=-0.5).aggregate_numpy_arrays_representation(np.array([[0.0, 0.5], [0.4, 0.9], [0.2, 0.6]]))

    print('A1=', a)
    print('A2=', b)
    print('A3=', c)
    print('A4=', d)
    print('A5=', e)
    print('A6=', f)
    print('A7=', g)
    print('A8=', h)
    print('A9=', i)
    print('A10=', j)

    print('A11=', k)
    print('A12=', l)
    print('A13=', u)
    print('A14=', x)
    # print('A15=', y)
    # print(yy)
    print('A16=', w)
    print('A17=', z)

    # Runtime warning
    # a = GMeanNumericallyImproved().aggregate_numpy_arrays_representation(np.array([[0.1, 0.7], [0.3, 0.6]]))
    # print(a)
    print(A14Aggregation(q=-1.5, s=-0.5, p=0.5).aggregate_numpy_arrays_representation(np.array([[0.,         0.44444444],
                                                                                                [0.,         0.44444444],
                                                                                               [0.,       0.44444444],
    [0.,         0.44444444],
    [0.,         0.42857143],
    [0.,         0.42857143],
    [0.2,        0.33333333],
    [0.,         0.42857143],
    [0.,         0.42857143],
    [0.,         0.42857143]])))
    print('sample for diagram')
    print(A11Aggregation(q=-2, s=1).aggregate_numpy_arrays_representation(np.array([[0, 0.28], [0, 0.28], [0, 0.57], [0, 0.14], [0, 0.42]])))

