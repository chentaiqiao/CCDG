from .q_learner import QLearner
from .ccdg_q_learner import CCDGQLearner



REGISTRY = {
    "q_learner": QLearner,
    "ccdg_q_learner": QLearner
}
