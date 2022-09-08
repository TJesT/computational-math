from typing import Callable
import numpy as np
from pytoolz.functional import pipe

FLOAT_CAP = 1e9

class NoRootsException(Exception):
    ...

def bisection_method(function: Callable[[float], float], 
                left_bound: float, 
                right_bound: float, 
                eps: float) -> float:
    sign_changed = function(left_bound) > function(right_bound)
    
    x = (right_bound + left_bound) / 2
    f_x = function(x)

    while abs(f_x) > eps:
        if sign_changed ^ (f_x < 0):
            left_bound = x
        else:
            right_bound = x
        
        x = (right_bound + left_bound) / 2
        f_x = function(x)

        if right_bound - left_bound <= eps:
            break
    # else:
    #     print('no roots was found')

    return x

class Polynomial:
    def __init__(self, eps: float, *coeffs: float):
        self.__eps = eps
        self.__accuracy = int(-np.log10(eps))
        self.__coeffs = pipe([
            lambda x: np.array(x, dtype=np.float64),
            np.ravel,
            lambda x: np.around(x, self.__accuracy),
            lambda x: np.trim_zeros(x, 'f')
        ], coeffs)

    @property
    def degree(self):
        return self.__coeffs.shape[0] - 1

    @property
    def diff(self):
        diff_coeffs = np.arange(self.degree, -1, -1, dtype=np.float64)
        diff_coeffs *= self.__coeffs
        
        return Polynomial(self.__eps, *diff_coeffs[:-1])

    def __call__(self, x: float) -> float:
        pows = np.arange(self.degree, -1, -1, dtype=np.float64)
        xes  = np.full(self.degree + 1, x, dtype=np.float64)
        
        result = np.around((self.__coeffs * xes ** pows).sum(), self.__accuracy)

        return 0 if abs(result) <= self.__eps else result

    @property
    def roots(self) -> list[float]:
        if self.degree <= 0:
            raise NoRootsException()
        if self.degree == 1:
            return np.around(-self.__coeffs[1]/self.__coeffs[0], self.__accuracy)
        if self.degree == 2:
            return self.__quadratic_roots()

        diff_roots = self.diff.roots

        intervals = self.__construct_intervals(diff_roots)

        # print()
        # print(self)
        roots = []
        for interval in intervals:
            ins = f'[{interval[0]}, {interval[1]}]'
            
            root = bisection_method(self, *interval, self.__eps)
            
            if abs(self(root)) < self.__eps:
                roots.append(np.around(root, self.__accuracy))
                # print(f'{root:.6f} in {ins}')
        
        return roots

    def __quadratic_roots(self) -> list[float]:
        if self.degree != 2:
            return []
        
        a, b, c = self.__coeffs

        D = b**2 - 4*a*c

        if D < 0:
            return []
        if D == 0:
            return [-b/2]
        
        return np.around(sorted([(-b-D**0.5)/2/a, (-b+D**0.5)/2/a]), self.__accuracy)

    def __construct_intervals(self, roots: list[float]) -> list[list[float, float]]:
        edges = [-FLOAT_CAP, *roots, +FLOAT_CAP]
        
        return [[l, b] for l, b in zip(edges, edges[1:])]
        
    def __str__(self):
        n = self.degree
        string_parts = [
            '{{coef:.{acc}g}} * x^{{pow}}'  \
                .format(acc=self.__accuracy) \
                .format(coef=coef, pow=n-i)
                for i, coef in enumerate(self.__coeffs) 
                    if abs(coef) > self.__eps
        ]

        return ' + '.join(string_parts) \
            .replace('x^1', 'x')         \
            .replace(' * x^0', '')        \
            .replace('+ -', '- ')

if __name__ == '__main__':
    p1 = Polynomial(1e-5, 0, 0, 0, -1, 2,  0)
    p2 = Polynomial(1e-5, 1, 1, -1, 2,  1)
    p3 = Polynomial(1e-5, 1, -1, -2, -1, 2,  1)
    p4 = p3.diff

    print(p1)
    print('roots are: ', p1.roots)
    print()

    print(p2)
    print('roots are: ', p2.roots)
    print()

    print(p3)
    print('roots are: ', p3.roots)
    print()

    print(p4)
    print('roots are: ', p4.roots)