"""Bounded largest-remainder allocation: exact budgets, no oversampling."""
import heapq
import numpy as np


def allocate_proportional(capacities, budget, minimum=0):
    capacities = np.asarray(capacities)
    if capacities.ndim != 1 or not np.issubdtype(capacities.dtype, np.integer) or np.any(capacities < 0):
        raise ValueError("capacities must be a one-dimensional list of nonnegative integers")
    if not isinstance(budget, (int, np.integer)) or budget < 0 or budget > int(capacities.sum()):
        raise ValueError("budget must be an integer within the available capacity")
    if minimum < 0:
        raise ValueError("minimum must be nonnegative")
    if budget == 0:
        return np.zeros(len(capacities), dtype=int)
    lower = np.minimum(capacities, minimum)
    # Eq. (4)'s lower bound may be impossible under a small fixed budget.
    if int(lower.sum()) > budget:
        lower = np.zeros_like(capacities)
    ideal = budget * capacities.astype(float) / capacities.sum()
    result = np.maximum(np.floor(ideal).astype(int), lower)
    result = np.minimum(result, capacities)
    difference = int(budget - result.sum())
    if difference > 0:
        heap = [(float(result[i] - ideal[i]), i) for i in range(len(result)) if result[i] < capacities[i]]
        heapq.heapify(heap)
        for _ in range(difference):
            _, i = heapq.heappop(heap)
            result[i] += 1
            if result[i] < capacities[i]:
                heapq.heappush(heap, (float(result[i] - ideal[i]), i))
    elif difference < 0:
        heap = [(float(ideal[i] - result[i]), i) for i in range(len(result)) if result[i] > lower[i]]
        heapq.heapify(heap)
        for _ in range(-difference):
            _, i = heapq.heappop(heap)
            result[i] -= 1
            if result[i] > lower[i]:
                heapq.heappush(heap, (float(ideal[i] - result[i]), i))
    return result
