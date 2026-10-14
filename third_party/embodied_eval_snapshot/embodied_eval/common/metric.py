import math


def mean(arr):
    return sum(arr) / len(arr)

def sample_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / (len(arr) - 1))

def mean_stderr(arr):
    return sample_stddev(arr) / math.sqrt(len(arr))

AGGREGATION = {
    'mean': mean
}
AGGREGATION_STD = {
    'mean': sample_stddev
}

def aggregation_for_metric(value, aggregation='mean'):
    agg_fn, agg_std = AGGREGATION[aggregation], AGGREGATION_STD[aggregation]
    
    aggregation_value = agg_fn(value)
    aggregation_value_stderr = agg_std(value)
    return (aggregation_value, aggregation_value_stderr)
