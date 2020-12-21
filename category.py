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
