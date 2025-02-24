import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import norm, expon
from scipy import integrate
from typing import Union, Tuple
from functools import partial


# # 封装的积分函数
# def integrate_function(a):
#     # 定义 integrand，利用 partial 绑定参数 'a'
#     def integrand(x, a):
#         return a * np.sin(x)

#     # 使用 partial 绑定 'a'
#     integrand_with_a = partial(integrand, a=a)

#     # 执行积分
#     result = integrate.quad_vec(integrand_with_a, 0, np.pi, workers=2)
#     return result


# 使用类来封装 integrand
# class Integrand:
#     def __init__(self, a):
#         self.a = a

#     def __call__(self, x):
#         return self.a * np.sin(x)

# # 封装的积分函数
# def integrate_function(a):
#     integrand = Integrand(a)
#     result = integrate.quad_vec(integrand, 0, np.pi, workers=2)
#     return result









class Distribution(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def get_name(self):
        pass
    
    @abstractmethod
    def cdf(self, t):
        """the cumulative distribution function
        Args:
            t: float, list, or numpy array, input values
        Returns:
            float, or numpy array, CDF values
        """
        pass

    @abstractmethod
    def pdf(self, t):
        """the probability densidty function
        Args:
            t: float, list, or numpy array, input values
        Returns:
            float, or numpy array, PDF values
        """
        pass

    @abstractmethod
    def random_sample(self, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
        """Generate random samples from the distribution
        
        Args:
            size: Number of samples to generate. Can be an integer or a tuple of integers
                 for multi-dimensional arrays.
            
        Returns:
            np.ndarray: Array of random samples with shape specified by size
        """
        pass



class NegExp(Distribution):
    def __init__(self, lmd=1.0):
        self.lmd = lmd
    
    def get_name(self):
        lmd = self.lmd
        return "NegExp-"+str(lmd)
    
    def cdf(self, t):
        if isinstance(t, list):
            t = np.array(t)
        return 1 - expon.cdf(-t, scale=1/self.lmd)

    def pdf(self, t):
        if isinstance(t, list):
            t = np.array(t)
        return expon.pdf(-t, scale=1/self.lmd)

    def random_sample(self, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
        return -np.random.exponential(1/self.lmd, size=size)

class NorMal(Distribution):  # always assume mean 0
    def __init__(self, std=1.0):
        self.std = std
    
    def get_name(self):
        std = self.std
        return "NorMal-"+str(std)
    
    def cdf(self, t):
        return norm.cdf(t, loc=0.0, scale=self.std)

    def pdf(self, t):
        return norm.pdf(t, loc=0.0, scale=self.std)

    def random_sample(self, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
        return np.random.normal(loc=0.0, scale=self.std, size=size)

class GumBel(Distribution):  # using Euler-Mascheroni Constant to get zero mean
    def __init__(self, eta=1.0):
        self.eta = eta
    
    def get_name(self):
        eta = self.eta
        return "GumBel-"+str(eta)
    
    def cdf(self, t):
        if isinstance(t, list):
            t = np.array(t)
        return np.exp(-np.exp(-(t/self.eta + np.euler_gamma)))

    def pdf(self, t):
        z = t/self.eta + np.euler_gamma
        return 1/self.eta * np.exp(-z - np.exp(-z))
        
    def random_sample(self, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
        return -self.eta*np.euler_gamma - self.eta * np.log(-np.log(np.random.uniform(size=size)))

class UniForm(Distribution):
    def __init__(self, delta=1.0):
        self.delta = delta
    
    def get_name(self):
        delta = self.delta
        return "UniForm-"+str(delta)
    
    def cdf(self, t):
        if isinstance(t, list):
            t = np.array(t)
        y = (t + self.delta)/(2 * self.delta)
        return np.clip(y, 0, 1)

    def pdf(self, t):
        if isinstance(t, (int, float)):
            return 1/(2*self.delta) if abs(t) <= self.delta else 0.0
        else:
            return np.where(abs(t) <= self.delta, 1/(2*self.delta), 0.0)
    
    def random_sample(self, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
        """Generate random samples from uniform distribution
        
        Args:
            size: Number of samples to generate. Can be an integer or a tuple of integers
                 for multi-dimensional arrays.
        """
        return np.random.uniform(low=-self.delta, high=self.delta, size=size)

class BimodalNormal(Distribution):
    def __init__(self, loc=3.0/np.sqrt(10), p=0.5, std=1.0/np.sqrt(10)):
        self.loc = loc
        self.p = p
        self.std = std
    
    def get_name(self):
        loc = self.loc
        p = self.p
        std = self.std
        return "BimodalNormal-"+str(loc)+"-"+str(p)+"-"+str(std)
    
    def cdf(self, t):
        if isinstance(t, list):
            t = np.array(t)
        return (self.p * norm.cdf(t, loc=-self.loc, scale=self.std) + 
                (1-self.p) * norm.cdf(t, loc=self.loc, scale=self.std)) 

    def pdf(self, t):
        return (self.p * norm.pdf(t, loc=-self.loc, scale=self.std) + 
                (1-self.p) * norm.pdf(t, loc=self.loc, scale=self.std)) 

    def random_sample(self, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
        normal_1 = np.random.normal(loc=-self.loc, scale=self.std, size=size)
        normal_2 = np.random.normal(loc=self.loc, scale=self.std, size=size)
        B = np.random.binomial(1, self.p, size=size)
        return B*normal_1 + (1-B)*normal_2