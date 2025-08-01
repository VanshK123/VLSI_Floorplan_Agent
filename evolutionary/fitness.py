"""Fitness functions and NSGA-II utilities for floorplanning."""

import numpy as np
from typing import List, Tuple, Dict, Any


def area(placement: Dict[str, np.ndarray]) -> float:
    """
    Calculate chip area from placement.
    
    Args:
        placement: Dictionary with 'x', 'y', 'width', 'height' arrays
        
    Returns:
        Total chip area (minimize)
    """
    x = placement['x']
    y = placement['y']
    width = placement['width']
    height = placement['height']
    
    # Calculate bounding box
    min_x = np.min(x)
    max_x = np.max(x + width)
    min_y = np.min(y)
    max_y = np.max(y + height)
    
    # Total area
    total_area = (max_x - min_x) * (max_y - min_y)
    
    # Add penalty for overlaps (simplified)
    overlap_penalty = _calculate_overlap_penalty(x, y, width, height)
    
    return total_area + overlap_penalty


def delay(placement: Dict[str, np.ndarray]) -> float:
    """
    Calculate critical path delay from placement.
    
    Args:
        placement: Dictionary with placement coordinates
        
    Returns:
        Critical path delay (minimize)
    """
    # This is a simplified delay calculation
    # In practice, this would use STA results
    
    x = placement['x']
    y = placement['y']
    
    # Calculate wirelength-based delay estimate
    # This assumes delay is proportional to wirelength
    wirelength = _calculate_total_wirelength(x, y)
    
    # Add placement-based delay factors
    placement_delay = _calculate_placement_delay(x, y)
    
    return wirelength + placement_delay


def _calculate_overlap_penalty(x: np.ndarray, y: np.ndarray, 
                             width: np.ndarray, height: np.ndarray) -> float:
    """Calculate penalty for cell overlaps."""
    penalty = 0.0
    num_cells = len(x)
    
    for i in range(num_cells):
        for j in range(i + 1, num_cells):
            # Check if rectangles overlap
            if (x[i] < x[j] + width[j] and x[i] + width[i] > x[j] and
                y[i] < y[j] + height[j] and y[i] + height[i] > y[j]):
                overlap_area = (min(x[i] + width[i], x[j] + width[j]) - 
                              max(x[i], x[j])) * (min(y[i] + height[i], y[j] + height[j]) - 
                                                  max(y[i], y[j]))
                penalty += overlap_area * 10  # High penalty for overlaps
    
    return penalty


def _calculate_total_wirelength(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate total wirelength (Manhattan distance)."""
    # Simplified wirelength calculation
    # In practice, this would use actual net connectivity
    
    # Assume cells are connected in a grid-like pattern
    total_wirelength = 0.0
    num_cells = len(x)
    
    for i in range(num_cells - 1):
        # Manhattan distance between adjacent cells
        wirelength = abs(x[i+1] - x[i]) + abs(y[i+1] - y[i])
        total_wirelength += wirelength
    
    return total_wirelength


def _calculate_placement_delay(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate delay based on placement characteristics."""
    # Factors that affect delay:
    # 1. Spread of cells (larger spread = longer wires)
    # 2. Density variations (congestion)
    
    # Calculate spread
    x_spread = np.max(x) - np.min(x)
    y_spread = np.max(y) - np.min(y)
    spread_factor = (x_spread + y_spread) / 1000  # Normalize
    
    # Calculate density (simplified)
    density_penalty = _calculate_density_penalty(x, y)
    
    return spread_factor + density_penalty


def _calculate_density_penalty(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate penalty for high-density regions."""
    # Divide placement area into grid and calculate density
    grid_size = 50
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    if x_max == x_min or y_max == y_min:
        return 0.0
    
    # Create density grid
    grid_x = int((x_max - x_min) / grid_size) + 1
    grid_y = int((y_max - y_min) / grid_size) + 1
    density_grid = np.zeros((grid_x, grid_y))
    
    # Count cells in each grid cell
    for cell_x, cell_y in zip(x, y):
        grid_i = int((cell_x - x_min) / grid_size)
        grid_j = int((cell_y - y_min) / grid_size)
        if 0 <= grid_i < grid_x and 0 <= grid_j < grid_y:
            density_grid[grid_i, grid_j] += 1
    
    # Penalty for high density regions
    max_density = np.max(density_grid)
    density_penalty = max_density * 0.1
    
    return density_penalty


def fast_non_dominated_sort(population: List[Any]) -> List[List[Any]]:
    """
    Fast non-dominated sorting for NSGA-II.
    
    Args:
        population: List of individuals with fitness values
        
    Returns:
        List of fronts (each front is a list of individuals)
    """
    fronts = []
    domination_count = {}
    dominated_solutions = {}
    
    # Initialize
    for p in population:
        domination_count[p] = 0
        dominated_solutions[p] = []
    
    # Calculate domination relationships
    for p in population:
        for q in population:
            if p != q:
                if _dominates(p, q):
                    dominated_solutions[p].append(q)
                elif _dominates(q, p):
                    domination_count[p] += 1
    
    # Find first front
    front = [p for p in population if domination_count[p] == 0]
    fronts.append(front)
    
    # Generate subsequent fronts
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
    
    return fronts


def _dominates(p: Any, q: Any) -> bool:
    """
    Check if individual p dominates individual q.
    
    Args:
        p, q: Individuals with fitness values
        
    Returns:
        True if p dominates q
    """
    # p dominates q if p is no worse than q in all objectives
    # and p is strictly better than q in at least one objective
    
    p_fitness = p.fitness.values
    q_fitness = q.fitness.values
    
    # Check if p is no worse than q in all objectives
    no_worse = all(p_fitness[i] <= q_fitness[i] for i in range(len(p_fitness)))
    
    # Check if p is strictly better than q in at least one objective
    strictly_better = any(p_fitness[i] < q_fitness[i] for i in range(len(p_fitness)))
    
    return no_worse and strictly_better


def calculate_crowding_distance(front: List[Any]) -> List[float]:
    """
    Calculate crowding distance for individuals in a front.
    
    Args:
        front: List of individuals in the same front
        
    Returns:
        List of crowding distances
    """
    if len(front) <= 2:
        return [float('inf')] * len(front)
    
    num_objectives = len(front[0].fitness.values)
    distances = [0.0] * len(front)
    
    # Calculate crowding distance for each objective
    for obj in range(num_objectives):
        # Sort front by objective value
        sorted_front = sorted(front, key=lambda x: x.fitness.values[obj])
        
        # Set boundary points to infinity
        distances[front.index(sorted_front[0])] = float('inf')
        distances[front.index(sorted_front[-1])] = float('inf')
        
        # Calculate distances for intermediate points
        obj_range = sorted_front[-1].fitness.values[obj] - sorted_front[0].fitness.values[obj]
        if obj_range == 0:
            continue
            
        for i in range(1, len(sorted_front) - 1):
            distance = (sorted_front[i + 1].fitness.values[obj] - 
                       sorted_front[i - 1].fitness.values[obj]) / obj_range
            distances[front.index(sorted_front[i])] += distance
    
    return distances


def calculate_hypervolume(pareto_front: List[Dict[str, float]], 
                         reference_point: List[float]) -> float:
    """
    Calculate hypervolume indicator for Pareto front quality.
    
    Args:
        pareto_front: List of solutions with fitness values
        reference_point: Reference point for hypervolume calculation
        
    Returns:
        Hypervolume value
    """
    if not pareto_front:
        return 0.0
    
    # Sort Pareto front by first objective
    sorted_front = sorted(pareto_front, key=lambda x: x['fitness'][0])
    
    # Calculate hypervolume using 2D approximation
    # For higher dimensions, use specialized libraries
    if len(reference_point) == 2:
        return _calculate_2d_hypervolume(sorted_front, reference_point)
    else:
        # Simplified calculation for higher dimensions
        return _calculate_simplified_hypervolume(sorted_front, reference_point)


def _calculate_2d_hypervolume(pareto_front: List[Dict[str, float]], 
                             reference_point: List[float]) -> float:
    """Calculate 2D hypervolume."""
    if not pareto_front:
        return 0.0
    
    hypervolume = 0.0
    prev_x = reference_point[0]
    
    for solution in pareto_front:
        x, y = solution['fitness']
        hypervolume += (prev_x - x) * (reference_point[1] - y)
        prev_x = x
    
    return hypervolume


def _calculate_simplified_hypervolume(pareto_front: List[Dict[str, float]], 
                                    reference_point: List[float]) -> float:
    """Simplified hypervolume calculation for higher dimensions."""
    # Use volume of convex hull approximation
    if len(pareto_front) < 3:
        return 0.0
    
    # Calculate volume of bounding box
    min_values = [min(sol['fitness'][i] for sol in pareto_front) for i in range(len(reference_point))]
    max_values = [max(sol['fitness'][i] for sol in pareto_front) for i in range(len(reference_point))]
    
    volume = 1.0
    for i in range(len(reference_point)):
        volume *= (reference_point[i] - min_values[i])
    
    return volume


def calculate_spread_metric(pareto_front: List[Dict[str, float]]) -> float:
    """
    Calculate spread metric for Pareto front diversity.
    
    Args:
        pareto_front: List of solutions
        
    Returns:
        Spread metric value
    """
    if len(pareto_front) < 2:
        return 0.0
    
    # Calculate distances between consecutive solutions
    distances = []
    sorted_front = sorted(pareto_front, key=lambda x: x['fitness'][0])
    
    for i in range(len(sorted_front) - 1):
        dist = np.linalg.norm(np.array(sorted_front[i]['fitness']) - 
                              np.array(sorted_front[i + 1]['fitness']))
        distances.append(dist)
    
    if not distances:
        return 0.0
    
    # Calculate spread metric
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    if mean_distance == 0:
        return 0.0
    
    return std_distance / mean_distance
