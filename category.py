from abc import ABCMeta, abstractmethod

def compose(f, g): return lambda x: f(g(x))

class Functor(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def lift(f): pass

    def push_through(self, f): return self.lift(f)(self)

class Monad(Functor):
    # join :: P P X -> P X
    @abstractmethod
    def join(self): pass

    # unit :: X -> P X
    @staticmethod
    @abstractmethod
    def unit(): pass

    # bind :: P X -> (X -> P Y) -> P Y
    def bind(self, f): return self.push_through(f).join()

    # revbind :: (X -> P Y) -> (P X -> P Y)
    @staticmethod
    def revbind(f): return lambda ta: ta.bind(f)

    # kleisli :: (X -> P Y) -> (Y -> P Z) -> (X -> P Z)
    @staticmethod
    def kleisli(f, g): return lambda a: f(a).bind(g)

    # liftM :: 
    @staticmethod
    def liftM(f): return lambda ma: ma.bind(lambda a: f(a).unit())

proj1 = bimap(ident, discard)
proj2 = compose(proj1, swap)

class Markov(Symmetric, Monoidal):
    @abstractmethod
    def copy(self): pass
    @abstractmethod
    def discard(self): pass

    def integrate(self, f):
        return bimap(ident, f)(self.copy())
    @abstractmethod
    def disintegrate(): pass

class Category(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, domain, codomain, morphism): pass

    @abstractmethod
    def compose(f, g): pass
    def precompose(f, g): return g.compose(f)
    @abstractmethod
    def ident(obj): pass

class Bifunctor(Category):
    @abstractmethod
    def bimap(f,g): pass

class Associative(Bifunctor):
    @abstractmethod
    def associate(f): pass
    @abstractmethod
    def disassociate(f): pass

class Monoidal(Associative):
    @abstractmethod
    def unit(): pasr
