from nlgeval import compute_metrics, compute_individual_metrics
import numpy as np

metrics_lt = {}

for i in range(20):
    metrics_dict = compute_metrics(hypothesis=f'gen_hyp/hyp_{i}.txt', references=[f'gen_hyp/ref_{i}.txt'])
    for name in metrics_dict:
        if name in metrics_lt:
            metrics_lt[name].append(metrics_dict[name])
        else:
            metrics_lt[name] = [metrics_dict[name]]

import scipy.stats
import math

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = math.sqrt(n) * se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

print('************ overall result ***************')
for name in metrics_lt:
    lt = metrics_lt[name]
    m, lb, hb = mean_confidence_interval(lt)
    print(f'name: {name}, mean: {m}, CI(95%): [{lb}, {hb}]')


