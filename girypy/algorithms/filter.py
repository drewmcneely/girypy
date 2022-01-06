def bayes_invert(probability, conditional):
    return (conditional * probability) / conditional.target

def update(former, dynamics, instrument, measurement):
    prior = dynamics @ former
    updater = bayes_invert(prior, instrument)
    return updater @ measurement
