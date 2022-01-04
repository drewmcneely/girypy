from abc import ABC, abstractmethod

class Category(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs): pass

    @abstractmethod
    def compose(self, other): pass
    def __matmul__(self,other): return self.compose(other)

    @classmethod
    @abstractmethod
    def identity(cls, obj): pass

    @property
    @abstractmethod
    def source(self): pass
    @property
    @abstractmethod
    def target(self): pass

class StrictMonoidal(Category):
    @abstractmethod
    def bimap(self, other): pass
    def __and__(self, other): return self.bimap(other)

    @staticmethod
    @abstractmethod
    def unit(): pass

    @staticmethod
    @abstractmethod
    def factor1(xy, x): pass
    @staticmethod
    @abstractmethod
    def factor2(xy, y): pass

class StrictSymmetric(StrictMonoidal):
    @classmethod
    @abstractmethod
    def swapper(cls, obj1, obj2): pass
    def swapped(self, obj1, obj2):
        return type(self).swapper(obj1, obj2) @ self

    @staticmethod
    @abstractmethod
    def factor(xy, x): pass
    def factor1(xy, x): return factor(xy, x)
    def factor2(xy, x): return factor(xy, x)

class StrictMarkov(StrictSymmetric):
    @classmethod
    @abstractmethod
    def copier(cls, obj): pass
    def copied(self):
        return type(self).copier(self.target) @ self

    @classmethod
    @abstractmethod
    def discarder(cls, obj): pass

    @abstractmethod
    def condition(self, x): pass
    def condition_l(self, x): return self.condition(x)
    def condition_r(self, y):
        xy = self.target
        x = type(self).factor(xy, y)
        return self.swapped(x,y).condition_l(y)
    def __truediv__(self, x):
        return self.condition_l(x)
    def __floordiv__(self, x):
        return self.condition_r(x)

    @classmethod
    def marginalizer_l(cls, obj1, obj2):
        return cls.identity(obj1) & cls.discarder(obj2)
    def margin_l(self, x):
        cls = type(self)
        y = cls.factor(self.target, x)
        return cls.marginalizer_l(x,y) @ self
    @classmethod
    def marginalizer_r(cls, obj1, obj2):
        return cls.marginalizer_l(obj2, obj1) @ cls.swapper(obj1,obj2)
    def margin_r(self, y):
        cls = type(self)
        x = cls.factor(self.target, y)
        return cls.marginalizer_r(x,y) @ self
