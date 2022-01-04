def integrator_r(self):
    cls = type(self)
    source = self.source
    return (cls.identity(source) & self) @ cls.copier(source)
def integrate_r(self, morphism):
    return morphism.integrator_r() @ self

def integrator_l(self):
    cls = type(self)
    l = self.target
    r = self.source
    return cls.swapper(r,l) @ self.integrator_r()
def integrate_l(self, morphism):
    return morphism.integrator_l() @ self

def __mul__(self, other):
    cls = type(self)
    unit = cls.unit()
    if other.source == cls.unit():
        assert other.target == self.source
        return other.integrate_l(self)
    elif self.source == cls.unit():
        assert self.target == other.source
        return self.integrate_r(other)
    else: raise ValueError

def bayes_invert(probability, conditional):
    return (conditional * probability) / conditional.target

def update(self, dynamics, instrument, measurement):
    prior = dynamics @ self
    updater = prior.bayes_invert(instrument)
    return updater @ measurement
