def bayes_invert(probability, conditional):
    return (conditional * probability) / conditional.target

def update(self, dynamics, instrument, measurement):
    prior = dynamics @ self
    updater = bayes_invert(prior, instrument)
    return updater @ measurement
