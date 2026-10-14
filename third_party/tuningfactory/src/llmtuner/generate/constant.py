from enum import Enum


class OutputType(Enum):
    Sequence = 'sequence'
    TokensProbabilities = 'tokens_probabilities'
    Embedding = 'embedding'
    RewardScores = 'reward_scores'
    Audio = 'audio'
