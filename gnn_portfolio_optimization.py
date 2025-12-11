#!/usr/bin/env python3
"""
GNN-Enhanced Stock Portfolio Optimization

This module implements a simple Graph Neural Network (GNN) to enhance stock feature
representations before applying SHLO and Hill Climbing optimization algorithms.

The GNN models relationships between stocks based on:
1. Category similarity (same market cap category)
2. Sector/Industry proximity (similar normalized features)
3. Performance correlation (similar fitness scores)

Innovation: Using GNN to learn enhanced stock embeddings that capture inter-stock
relationships, which are then used to improve the objective function in portfolio optimization.

Author: Research Implementation
Date: 2024
"""

import numpy as np
import pandas as pd
import random
import time
import copy
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


#Graph GNN implementation

class SimpleGNN:
    """
    A minimal Graph Neural Network implementation for stock relationship modeling.
    
    This GNN:
    1. Builds a graph where nodes are stocks and edges connect similar stocks
    2. Uses message passing to aggregate neighbor information
    3. Produces enhanced feature representations for each stock
    
    The enhanced features capture not just individual stock metrics but also
    the stock's relationship to other stocks in the universe.
    """
    
    def __init__(self, input_dim, hidden_dim=16, output_dim=8, num_layers=2):
        """
        Initialize the GNN.
        
        Args:
            input_dim: Number of input features per stock
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of message passing layers
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Initialize weights with Xavier initialization
        self.weights = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(len(dims) - 1):
            limit = np.sqrt(6.0 / (dims[i] + dims[i+1]))  # Xavier initialization
            W = np.random.uniform(-limit, limit, (dims[i], dims[i+1]))
            self.weights.append(W)
        
        print(f"[GNN] Initialized with {num_layers} layers: {dims}")
    
    def build_adjacency_matrix(self, features, category, threshold=0.3):
        """
        Build adjacency matrix based on stock similarity.
        
        Stocks are connected if:
        1. They are in the same category (market cap)
        2. Their normalized features are similar (cosine similarity > threshold)
        
        Args:
            features: Stock feature matrix (num_stocks x num_features)
            category: Category labels for each stock
            threshold: Similarity threshold for edge creation
            
        Returns:
            Adjacency matrix (num_stocks x num_stocks)
        """
        num_stocks = features.shape[0]
        adj = np.zeros((num_stocks, num_stocks))
        
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized_features = features / norms
        similarity = np.dot(normalized_features, normalized_features.T)
        
        for i in range(num_stocks):
            for j in range(i + 1, num_stocks):
                same_category = category[i] == category[j]
                high_similarity = similarity[i, j] > threshold
                if same_category or high_similarity:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        adj += np.eye(num_stocks)  # Self-loops
        
        # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        degree = np.sum(adj, axis=1)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        adj_normalized = D_inv_sqrt @ adj @ D_inv_sqrt
        
        return adj_normalized
    
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def forward(self, features, adj):
        """
        Forward pass through the GNN.
        
        Args:
            features: Input feature matrix (num_stocks x input_dim)
            adj: Normalized adjacency matrix (num_stocks x num_stocks)
            
        Returns:
            Enhanced feature embeddings (num_stocks x output_dim)
        """
        h = features
        
        for i, W in enumerate(self.weights):
            h = adj @ h  # Aggregate neighbor features
            h = h @ W    # Linear transformation
            if i < len(self.weights) - 1:
                h = self.relu(h)
        
        return h
    
    def get_enhanced_scores(self, embeddings, original_scores):
        """
        Combine GNN embeddings with original scores to get enhanced scores.
        
        Args:
            embeddings: GNN output embeddings (num_stocks x output_dim)
            original_scores: Original normalized scores (num_stocks x 3)
            
        Returns:
            Enhanced scores (num_stocks x 3)
        """
        embedding_score = np.mean(embeddings, axis=1)
        
        # Min-max normalization
        if embedding_score.max() != embedding_score.min():
            embedding_score = (embedding_score - embedding_score.min()) / (embedding_score.max() - embedding_score.min())
        else:
            embedding_score = np.ones_like(embedding_score) * 0.5
        
        # Combine: 80% original + 20% GNN
        gnn_weight = 0.2
        enhanced_scores = original_scores.copy()
        
        for i in range(original_scores.shape[1]):
            enhanced_scores[:, i] = (1 - gnn_weight) * original_scores[:, i] + gnn_weight * embedding_score
        
        return enhanced_scores


def prepare_gnn_features(stocks_df):
    """
    Prepare feature matrix for GNN from stock dataframe.
    
    Args:
        stocks_df: DataFrame with stock data
        
    Returns:
        features: Feature matrix (num_stocks x num_features)
        categories: Category array
        original_scores: Original normalized scores
    """
    feature_cols = [
        'normalized_fitness_score',
        'normalized_percent_change_from_instrinsic_value',
        'normalized_Rev_Gr_Next_Y'
    ]
    features = stocks_df[feature_cols].fillna(0).values
    
    # Additional normalized features
    prices = stocks_df['Stock Price'].values.astype(float)
    prices_norm = (prices - prices.min()) / (prices.max() - prices.min() + 1e-8)
    
    fitness = stocks_df['Combined_Fitness_Score'].fillna(0).values.astype(float)
    fitness_norm = (fitness - fitness.min()) / (fitness.max() - fitness.min() + 1e-8)
    
    # Category one-hot encoding
    categories = stocks_df['Category'].values.astype(int)
    category_onehot = np.zeros((len(categories), 3))
    for i, cat in enumerate(categories):
        if 1 <= cat <= 3:
            category_onehot[i, cat - 1] = 1
    
    # Feature matrix: [3 normalized + 1 price + 1 fitness + 3 category] = 8 dims
    all_features = np.column_stack([
        features,
        prices_norm.reshape(-1, 1),
        fitness_norm.reshape(-1, 1),
        category_onehot
    ])
    
    return all_features, categories, features


#GNN- With SHLO algrotihm

class BudgetAllocation:
    def __init__(self, total_budget, upper_budget_limit, lower_budget_limit):
        self.total_budget = total_budget
        self.upper_budget_limit = upper_budget_limit
        self.lower_budget_limit = lower_budget_limit

    def calculate_budget_allocation(self, stock_price, min_quantity, max_quantity):
        upper_budget = int(self.total_budget * self.upper_budget_limit)
        lower_budget = int(self.total_budget * self.lower_budget_limit)
        
        for quantity in range(min_quantity, max_quantity + 1):
            budget_used = stock_price * quantity
            if lower_budget <= budget_used <= upper_budget:
                return quantity, budget_used, self.total_budget - budget_used, True
        
        return None, None, None, False


class Node:
    def __init__(self, solution=None):
        self.sol = solution if solution is not None else []
        self.fitness = 0
        self.ikd = [[] for _ in range(6)]  # Individual Knowledge Database


def run_gnn_enhanced_shlo(stocks_df, portfolio_size, large_cap_count, mid_cap_count,
                          total_budget, upper_budget_limit, lower_budget_limit,
                          weight_fitness, weight_percent_change, weight_rev_growth, 
                          weight_normalized_budget, epochs=100, pop_size=50):
    """
    Run SHLO algorithm with GNN-enhanced stock features.
    
    Args:
        stocks_df: DataFrame with stock data
        portfolio_size: Number of stocks in portfolio
        large_cap_count: Number of large-cap stocks
        mid_cap_count: Number of mid-cap stocks
        total_budget: Total investment budget
        upper_budget_limit: Max percentage per stock
        lower_budget_limit: Min percentage per stock
        weight_*: Objective function weights
        epochs: Number of optimization epochs
        pop_size: Population size
        
    Returns:
        best_portfolio: List of selected stocks
        best_fitness: Best objective function value
        total_budget_used: Total budget utilized
        time_taken: Execution time
        gnn_embeddings: GNN output embeddings
    """
    print("\n" + "="*60)
    print("GNN-ENHANCED SHLO ALGORITHM")
    print("="*60)
    
    start_time = time.time()
    
    # Step 1: Prepare GNN features
    print("\n[Step 1] Preparing GNN features...")
    gnn_features, categories, original_scores = prepare_gnn_features(stocks_df)
    print(f"  - Feature matrix shape: {gnn_features.shape}")
    
    # Step 2: Initialize and run GNN
    print("\n[Step 2] Running GNN forward pass...")
    gnn = SimpleGNN(input_dim=gnn_features.shape[1], hidden_dim=16, output_dim=8, num_layers=2)
    adj = gnn.build_adjacency_matrix(gnn_features, categories, threshold=0.3)
    print(f"  - Adjacency matrix density: {np.sum(adj > 0) / (adj.shape[0] * adj.shape[1]):.4f}")
    
    embeddings = gnn.forward(gnn_features, adj)
    print(f"  - Embedding shape: {embeddings.shape}")
    
    # Step 3: Get enhanced scores
    print("\n[Step 3] Computing enhanced scores...")
    enhanced_scores = gnn.get_enhanced_scores(embeddings, original_scores)
    
    # Update stocks dataframe with enhanced scores
    stocks_enhanced = stocks_df.copy()
    stocks_enhanced['gnn_fitness_score'] = enhanced_scores[:, 0]
    stocks_enhanced['gnn_percent_change'] = enhanced_scores[:, 1]
    stocks_enhanced['gnn_rev_growth'] = enhanced_scores[:, 2]
    
    # Step 4: Run SHLO with enhanced features
    print("\n[Step 4] Running SHLO optimization...")
    
    num_stocks = len(stocks_enhanced)
    small_cap_count = portfolio_size - large_cap_count - mid_cap_count
    
    budget_allocator = BudgetAllocation(total_budget, upper_budget_limit, lower_budget_limit)
    
    population = []
    for _ in range(pop_size):
        node = Node()
        node.sol = [0] * num_stocks
        selected = random.sample(range(num_stocks), portfolio_size)
        for i in selected:
            node.sol[i] = 1
        population.append(node)
    
    skd = [[0] * num_stocks for _ in range(10)]  # Social Knowledge Database
    for k in range(1, 10):
        selected = random.sample(range(num_stocks), portfolio_size)
        for i in selected:
            skd[k][i] = 1
    
    for node in population:  # Individual Knowledge Database
        for k in range(1, 6):
            node.ikd[k] = [0] * num_stocks
            selected = random.sample(range(num_stocks), portfolio_size)
            for i in selected:
                node.ikd[k][i] = 1
    
    def check_category_constraint(selected_indices):
        large = sum(1 for idx in selected_indices if stocks_enhanced.iloc[idx]['Category'] == 1)
        mid = sum(1 for idx in selected_indices if stocks_enhanced.iloc[idx]['Category'] == 2)
        small = sum(1 for idx in selected_indices if stocks_enhanced.iloc[idx]['Category'] == 3)
        return large == large_cap_count and mid == mid_cap_count and small == small_cap_count
    
    def check_constraints(solution):
        selected_indices = [i for i, x in enumerate(solution) if x == 1]
        if len(selected_indices) != portfolio_size:
            return False
        if not check_category_constraint(selected_indices):
            return False
        
        total_budget_used = 0
        for idx in selected_indices:
            stock = stocks_enhanced.iloc[idx]
            _, budget_used, _, success = budget_allocator.calculate_budget_allocation(
                stock['Stock Price'], int(stock['Lower_limit']), int(stock['Upper_limit'])
            )
            if not success:
                return False
            total_budget_used += budget_used
        
        if total_budget_used > total_budget:
            return False
        return True
    
    def calculate_fitness(solution):
        if not check_constraints(solution):
            return 0
        
        selected_indices = [i for i, x in enumerate(solution) if x == 1]
        total_obj = 0
        total_budget_used = 0
        
        for idx in selected_indices:
            stock = stocks_enhanced.iloc[idx]
            _, budget_used, _, success = budget_allocator.calculate_budget_allocation(
                stock['Stock Price'], int(stock['Lower_limit']), int(stock['Upper_limit'])
            )
            if success:
                total_budget_used += budget_used
                obj_value = (
                    weight_fitness * stock['gnn_fitness_score'] +
                    weight_percent_change * stock['gnn_percent_change'] +
                    weight_rev_growth * stock['gnn_rev_growth'] +
                    weight_normalized_budget * (total_budget_used / total_budget)
                )
                total_obj += obj_value
        
        return total_obj
    
    best_fitness_history = []
    
    for epoch in range(epochs):
        iteration = epoch / epochs
        probabilities = [0.4, 0.3, 0.3]
        perturbation_rate = max(0.2, 0.5 * (1 - iteration))
        
        for node in population:
            original_sol = node.sol.copy()
            original_fitness = node.fitness
            
            for j in range(len(node.sol)):
                rand_choice = random.random()
                
                if rand_choice <= probabilities[0]:
                    if random.random() < perturbation_rate:
                        node.sol[j] = 1 - node.sol[j]
                elif rand_choice <= probabilities[0] + probabilities[1]:
                    w1 = 1 - iteration
                    ikd_value = node.ikd[random.randint(1, 5)][j]
                    skd_value = skd[random.randint(1, 9)][j]
                    node.sol[j] = ikd_value if w1 > random.random() else skd_value
                else:
                    node.sol[j] = skd[random.randint(1, 9)][j]
            
            while sum(node.sol) != portfolio_size:
                if sum(node.sol) > portfolio_size:
                    indices = [i for i, x in enumerate(node.sol) if x == 1]
                    node.sol[random.choice(indices)] = 0
                else:
                    indices = [i for i, x in enumerate(node.sol) if x == 0]
                    node.sol[random.choice(indices)] = 1
            
            if not check_constraints(node.sol):
                node.sol = original_sol
                node.fitness = original_fitness
            else:
                new_fitness = calculate_fitness(node.sol)
                if new_fitness > node.fitness:
                    node.fitness = new_fitness
                else:
                    node.sol = original_sol
                    node.fitness = original_fitness
        
        population.sort(key=lambda x: x.fitness, reverse=True)
        best_fitness_history.append(population[0].fitness)
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}, Best Fitness: {population[0].fitness:.4f}")
    
    best_node = population[0]
    selected_indices = [i for i, x in enumerate(best_node.sol) if x == 1]
    
    portfolio = []
    total_budget_used = 0
    for idx in selected_indices:
        stock = stocks_enhanced.iloc[idx]
        quantity, budget_used, _, _ = budget_allocator.calculate_budget_allocation(
            stock['Stock Price'], int(stock['Lower_limit']), int(stock['Upper_limit'])
        )
        total_budget_used += budget_used
        portfolio.append([
            int(stock['Stock_ID']),
            stock['Symbol'],
            stock['Company Name'],
            stock['Stock Price'],
            int(stock['Category']),
            quantity
        ])
    
    time_taken = time.time() - start_time
    
    print(f"\n[GNN-SHLO] Optimization completed in {time_taken:.2f} seconds")
    
    return portfolio, best_node.fitness, total_budget_used, time_taken, embeddings


#GNN- Enhanced with Hill Climbing Algorithm

def run_gnn_enhanced_hill_climbing(stocks_df, portfolio_size, large_cap_count, mid_cap_count,
                                    total_budget, upper_budget_limit, lower_budget_limit,
                                    weight_fitness, weight_percent_change, weight_rev_growth,
                                    weight_normalized_budget, iterations=1000):
    """
    Run Hill Climbing algorithm with GNN-enhanced stock features.
    """
    print("\n" + "="*60)
    print("GNN-ENHANCED HILL CLIMBING ALGORITHM")
    print("="*60)
    
    start_time = time.time()
    
    # Step 1: Prepare GNN features
    print("\n[Step 1] Preparing GNN features...")
    gnn_features, categories, original_scores = prepare_gnn_features(stocks_df)
    
    # Step 2: Initialize and run GNN
    print("\n[Step 2] Running GNN forward pass...")
    gnn = SimpleGNN(input_dim=gnn_features.shape[1], hidden_dim=16, output_dim=8, num_layers=2)
    adj = gnn.build_adjacency_matrix(gnn_features, categories, threshold=0.3)
    embeddings = gnn.forward(gnn_features, adj)
    
    # Step 3: Get enhanced scores
    print("\n[Step 3] Computing enhanced scores...")
    enhanced_scores = gnn.get_enhanced_scores(embeddings, original_scores)
    
    # Update stocks with enhanced scores
    stocks_enhanced = stocks_df.copy()
    stocks_enhanced['gnn_fitness_score'] = enhanced_scores[:, 0]
    stocks_enhanced['gnn_percent_change'] = enhanced_scores[:, 1]
    stocks_enhanced['gnn_rev_growth'] = enhanced_scores[:, 2]
    
    list_data = stocks_enhanced.values.tolist()
    num_stocks = len(list_data)
    small_cap_count = portfolio_size - large_cap_count - mid_cap_count
    
    COL_STOCK_ID = 0
    COL_SYMBOL = 1
    COL_NAME = 2
    COL_PRICE = 3
    COL_CATEGORY = 4
    COL_LOWER = 9
    COL_UPPER = 10
    COL_GNN_FITNESS = -3
    COL_GNN_PERCENT = -2
    COL_GNN_REV = -1
    
    def calculate_budget(stock_price, min_qty, max_qty):
        upper_budget = int(total_budget * upper_budget_limit)
        lower_budget = int(total_budget * lower_budget_limit)
        
        for qty in range(int(min_qty), int(max_qty) + 1):
            budget_used = stock_price * qty
            if lower_budget <= budget_used <= upper_budget:
                return qty, budget_used, True
        return None, None, False
    
    def check_category(selected):
        large = sum(1 for s in selected if s[COL_CATEGORY] == 1)
        mid = sum(1 for s in selected if s[COL_CATEGORY] == 2)
        small = sum(1 for s in selected if s[COL_CATEGORY] == 3)
        return large == large_cap_count and mid == mid_cap_count and small == small_cap_count
    
    def calculate_objective(selected, quantities):
        total_obj = 0
        total_used = 0
        
        for i, stock in enumerate(selected):
            qty = quantities[i]
            budget_used = stock[COL_PRICE] * qty
            total_used += budget_used
            
            obj_value = (
                weight_fitness * stock[COL_GNN_FITNESS] +
                weight_percent_change * stock[COL_GNN_PERCENT] +
                weight_rev_growth * stock[COL_GNN_REV] +
                weight_normalized_budget * (total_used / total_budget)
            )
            total_obj += obj_value
        
        return total_obj, total_used
    
    # Generate initial feasible solution
    print("\n[Step 4] Generating initial solution...")
    max_attempts = 10000
    attempt = 0
    
    while attempt < max_attempts:
        selected = random.sample(list_data, portfolio_size)
        
        if not check_category(selected):
            attempt += 1
            continue
        
        quantities = []
        valid = True
        total_used = 0
        
        for stock in selected:
            qty, budget, success = calculate_budget(stock[COL_PRICE], stock[COL_LOWER], stock[COL_UPPER])
            if not success:
                valid = False
                break
            quantities.append(qty)
            total_used += budget
        
        if valid and total_used <= total_budget:
            break
        attempt += 1
    
    if attempt >= max_attempts:
        print("  Warning: Could not find feasible initial solution")
        return None, 0, 0, 0, None
    
    current_portfolio = selected
    current_quantities = quantities
    current_obj, current_budget = calculate_objective(current_portfolio, current_quantities)
    
    best_portfolio = copy.deepcopy(current_portfolio)
    best_quantities = current_quantities.copy()
    best_obj = current_obj
    best_budget = current_budget
    
    print("\n[Step 5] Running Hill Climbing optimization...")
    
    for i in range(iterations):
        neighbor = copy.deepcopy(current_portfolio)
        neighbor_qty = current_quantities.copy()
        
        idx1, idx2 = random.sample(range(portfolio_size), 2)
        cat1 = neighbor[idx1][COL_CATEGORY]
        cat2 = neighbor[idx2][COL_CATEGORY]
        
        candidates1 = [s for s in list_data if s[COL_CATEGORY] == cat1 and s not in neighbor]
        candidates2 = [s for s in list_data if s[COL_CATEGORY] == cat2 and s not in neighbor]
        
        if not candidates1 or not candidates2:
            continue
        
        new_stock1 = random.choice(candidates1)
        new_stock2 = random.choice(candidates2)
        
        qty1, _, success1 = calculate_budget(new_stock1[COL_PRICE], new_stock1[COL_LOWER], new_stock1[COL_UPPER])
        qty2, _, success2 = calculate_budget(new_stock2[COL_PRICE], new_stock2[COL_LOWER], new_stock2[COL_UPPER])
        
        if not success1 or not success2:
            continue
        
        neighbor[idx1] = new_stock1
        neighbor[idx2] = new_stock2
        neighbor_qty[idx1] = qty1
        neighbor_qty[idx2] = qty2
        
        neighbor_obj, neighbor_budget = calculate_objective(neighbor, neighbor_qty)
        
        if neighbor_budget <= total_budget and neighbor_obj > current_obj:
            current_portfolio = neighbor
            current_quantities = neighbor_qty
            current_obj = neighbor_obj
            current_budget = neighbor_budget
            
            if current_obj > best_obj:
                best_portfolio = copy.deepcopy(current_portfolio)
                best_quantities = current_quantities.copy()
                best_obj = current_obj
                best_budget = current_budget
        
        if i % 200 == 0:
            print(f"  Iteration {i}, Best Objective: {best_obj:.4f}")
    
    portfolio = []
    for i, stock in enumerate(best_portfolio):
        portfolio.append([
            int(stock[COL_STOCK_ID]),
            stock[COL_SYMBOL],
            stock[COL_NAME],
            stock[COL_PRICE],
            int(stock[COL_CATEGORY]),
            best_quantities[i]
        ])
    
    time_taken = time.time() - start_time
    
    print(f"\n[GNN-HC] Optimization completed in {time_taken:.2f} seconds")
    
    return portfolio, best_obj, best_budget, time_taken, embeddings


#Baseline Algorithms for Comparison

def run_baseline_shlo(stocks_df, portfolio_size, large_cap_count, mid_cap_count,
                      total_budget, upper_budget_limit, lower_budget_limit,
                      weight_fitness, weight_percent_change, weight_rev_growth,
                      weight_normalized_budget, epochs=100, pop_size=50):
    """Run baseline SHLO without GNN enhancement."""
    print("\n" + "="*60)
    print("BASELINE SHLO ALGORITHM (No GNN)")
    print("="*60)
    
    start_time = time.time()
    
    num_stocks = len(stocks_df)
    small_cap_count = portfolio_size - large_cap_count - mid_cap_count
    
    budget_allocator = BudgetAllocation(total_budget, upper_budget_limit, lower_budget_limit)
    
    population = []
    for _ in range(pop_size):
        node = Node()
        node.sol = [0] * num_stocks
        selected = random.sample(range(num_stocks), portfolio_size)
        for i in selected:
            node.sol[i] = 1
        population.append(node)
    
    skd = [[0] * num_stocks for _ in range(10)]
    for k in range(1, 10):
        selected = random.sample(range(num_stocks), portfolio_size)
        for i in selected:
            skd[k][i] = 1
    
    for node in population:
        for k in range(1, 6):
            node.ikd[k] = [0] * num_stocks
            selected = random.sample(range(num_stocks), portfolio_size)
            for i in selected:
                node.ikd[k][i] = 1
    
    def check_category_constraint(selected_indices):
        large = sum(1 for idx in selected_indices if stocks_df.iloc[idx]['Category'] == 1)
        mid = sum(1 for idx in selected_indices if stocks_df.iloc[idx]['Category'] == 2)
        small = sum(1 for idx in selected_indices if stocks_df.iloc[idx]['Category'] == 3)
        return large == large_cap_count and mid == mid_cap_count and small == small_cap_count
    
    def check_constraints(solution):
        selected_indices = [i for i, x in enumerate(solution) if x == 1]
        if len(selected_indices) != portfolio_size:
            return False
        if not check_category_constraint(selected_indices):
            return False
        
        total_budget_used = 0
        for idx in selected_indices:
            stock = stocks_df.iloc[idx]
            _, budget_used, _, success = budget_allocator.calculate_budget_allocation(
                stock['Stock Price'], int(stock['Lower_limit']), int(stock['Upper_limit'])
            )
            if not success:
                return False
            total_budget_used += budget_used
        
        if total_budget_used > total_budget:
            return False
        return True
    
    def calculate_fitness(solution):
        if not check_constraints(solution):
            return 0
        
        selected_indices = [i for i, x in enumerate(solution) if x == 1]
        total_obj = 0
        total_budget_used = 0
        
        for idx in selected_indices:
            stock = stocks_df.iloc[idx]
            _, budget_used, _, success = budget_allocator.calculate_budget_allocation(
                stock['Stock Price'], int(stock['Lower_limit']), int(stock['Upper_limit'])
            )
            if success:
                total_budget_used += budget_used
                fitness_score = stock['normalized_fitness_score'] if pd.notna(stock['normalized_fitness_score']) else 0
                percent_change = stock['normalized_percent_change_from_instrinsic_value'] if pd.notna(stock['normalized_percent_change_from_instrinsic_value']) else 0
                rev_growth = stock['normalized_Rev_Gr_Next_Y'] if pd.notna(stock['normalized_Rev_Gr_Next_Y']) else 0
                
                obj_value = (
                    weight_fitness * fitness_score +
                    weight_percent_change * percent_change +
                    weight_rev_growth * rev_growth +
                    weight_normalized_budget * (total_budget_used / total_budget)
                )
                total_obj += obj_value
        
        return total_obj
    
    for epoch in range(epochs):
        iteration = epoch / epochs
        probabilities = [0.4, 0.3, 0.3]
        perturbation_rate = max(0.2, 0.5 * (1 - iteration))
        
        for node in population:
            original_sol = node.sol.copy()
            original_fitness = node.fitness
            
            for j in range(len(node.sol)):
                rand_choice = random.random()
                
                if rand_choice <= probabilities[0]:
                    if random.random() < perturbation_rate:
                        node.sol[j] = 1 - node.sol[j]
                elif rand_choice <= probabilities[0] + probabilities[1]:
                    w1 = 1 - iteration
                    ikd_value = node.ikd[random.randint(1, 5)][j]
                    skd_value = skd[random.randint(1, 9)][j]
                    node.sol[j] = ikd_value if w1 > random.random() else skd_value
                else:
                    node.sol[j] = skd[random.randint(1, 9)][j]
            
            while sum(node.sol) != portfolio_size:
                if sum(node.sol) > portfolio_size:
                    indices = [i for i, x in enumerate(node.sol) if x == 1]
                    node.sol[random.choice(indices)] = 0
                else:
                    indices = [i for i, x in enumerate(node.sol) if x == 0]
                    node.sol[random.choice(indices)] = 1
            
            if not check_constraints(node.sol):
                node.sol = original_sol
                node.fitness = original_fitness
            else:
                new_fitness = calculate_fitness(node.sol)
                if new_fitness > node.fitness:
                    node.fitness = new_fitness
                else:
                    node.sol = original_sol
                    node.fitness = original_fitness
        
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}, Best Fitness: {population[0].fitness:.4f}")
    
    best_node = population[0]
    selected_indices = [i for i, x in enumerate(best_node.sol) if x == 1]
    
    portfolio = []
    total_budget_used = 0
    for idx in selected_indices:
        stock = stocks_df.iloc[idx]
        quantity, budget_used, _, _ = budget_allocator.calculate_budget_allocation(
            stock['Stock Price'], int(stock['Lower_limit']), int(stock['Upper_limit'])
        )
        total_budget_used += budget_used
        portfolio.append([
            int(stock['Stock_ID']),
            stock['Symbol'],
            stock['Company Name'],
            stock['Stock Price'],
            int(stock['Category']),
            quantity
        ])
    
    time_taken = time.time() - start_time
    print(f"\n[Baseline SHLO] Completed in {time_taken:.2f} seconds")
    
    return portfolio, best_node.fitness, total_budget_used, time_taken


def run_baseline_hill_climbing(stocks_df, portfolio_size, large_cap_count, mid_cap_count,
                                total_budget, upper_budget_limit, lower_budget_limit,
                                weight_fitness, weight_percent_change, weight_rev_growth,
                                weight_normalized_budget, iterations=1000):
    """Run baseline Hill Climbing without GNN enhancement."""
    print("\n" + "="*60)
    print("BASELINE HILL CLIMBING ALGORITHM (No GNN)")
    print("="*60)
    
    start_time = time.time()
    
    list_data = stocks_df.values.tolist()
    num_stocks = len(list_data)
    small_cap_count = portfolio_size - large_cap_count - mid_cap_count
    
    COL_STOCK_ID, COL_SYMBOL, COL_NAME, COL_PRICE, COL_CATEGORY = 0, 1, 2, 3, 4
    COL_LOWER, COL_UPPER, COL_FITNESS, COL_PERCENT, COL_REV = 9, 10, 11, 12, 13
    
    def calculate_budget(stock_price, min_qty, max_qty):
        upper_budget = int(total_budget * upper_budget_limit)
        lower_budget = int(total_budget * lower_budget_limit)
        for qty in range(int(min_qty), int(max_qty) + 1):
            budget_used = stock_price * qty
            if lower_budget <= budget_used <= upper_budget:
                return qty, budget_used, True
        return None, None, False
    
    def check_category(selected):
        large = sum(1 for s in selected if s[COL_CATEGORY] == 1)
        mid = sum(1 for s in selected if s[COL_CATEGORY] == 2)
        small = sum(1 for s in selected if s[COL_CATEGORY] == 3)
        return large == large_cap_count and mid == mid_cap_count and small == small_cap_count
    
    def calculate_objective(selected, quantities):
        total_obj = 0
        total_used = 0
        for i, stock in enumerate(selected):
            qty = quantities[i]
            budget_used = stock[COL_PRICE] * qty
            total_used += budget_used
            
            fitness = stock[COL_FITNESS] if not pd.isna(stock[COL_FITNESS]) else 0
            percent = stock[COL_PERCENT] if not pd.isna(stock[COL_PERCENT]) else 0
            rev = stock[COL_REV] if not pd.isna(stock[COL_REV]) else 0
            
            obj_value = (
                weight_fitness * fitness +
                weight_percent_change * percent +
                weight_rev_growth * rev +
                weight_normalized_budget * (total_used / total_budget)
            )
            total_obj += obj_value
        return total_obj, total_used
    
    print("\n  Generating initial solution...")
    max_attempts = 10000
    attempt = 0
    
    while attempt < max_attempts:
        selected = random.sample(list_data, portfolio_size)
        
        if not check_category(selected):
            attempt += 1
            continue
        
        quantities = []
        valid = True
        total_used = 0
        
        for stock in selected:
            qty, budget, success = calculate_budget(stock[COL_PRICE], stock[COL_LOWER], stock[COL_UPPER])
            if not success:
                valid = False
                break
            quantities.append(qty)
            total_used += budget
        
        if valid and total_used <= total_budget:
            break
        attempt += 1
    
    if attempt >= max_attempts:
        print("  Warning: Could not find feasible initial solution")
        return None, 0, 0, 0
    
    current_portfolio = selected
    current_quantities = quantities
    current_obj, current_budget = calculate_objective(current_portfolio, current_quantities)
    
    best_portfolio = copy.deepcopy(current_portfolio)
    best_quantities = current_quantities.copy()
    best_obj = current_obj
    best_budget = current_budget
    
    print("\n  Running optimization...")
    for i in range(iterations):
        neighbor = copy.deepcopy(current_portfolio)
        neighbor_qty = current_quantities.copy()
        
        idx1, idx2 = random.sample(range(portfolio_size), 2)
        cat1, cat2 = neighbor[idx1][COL_CATEGORY], neighbor[idx2][COL_CATEGORY]
        
        candidates1 = [s for s in list_data if s[COL_CATEGORY] == cat1 and s not in neighbor]
        candidates2 = [s for s in list_data if s[COL_CATEGORY] == cat2 and s not in neighbor]
        
        if not candidates1 or not candidates2:
            continue
        
        new_stock1, new_stock2 = random.choice(candidates1), random.choice(candidates2)
        qty1, _, success1 = calculate_budget(new_stock1[COL_PRICE], new_stock1[COL_LOWER], new_stock1[COL_UPPER])
        qty2, _, success2 = calculate_budget(new_stock2[COL_PRICE], new_stock2[COL_LOWER], new_stock2[COL_UPPER])
        
        if not success1 or not success2:
            continue
        
        neighbor[idx1], neighbor[idx2] = new_stock1, new_stock2
        neighbor_qty[idx1], neighbor_qty[idx2] = qty1, qty2
        neighbor_obj, neighbor_budget = calculate_objective(neighbor, neighbor_qty)
        
        if neighbor_budget <= total_budget and neighbor_obj > current_obj:
            current_portfolio, current_quantities = neighbor, neighbor_qty
            current_obj, current_budget = neighbor_obj, neighbor_budget
            
            if current_obj > best_obj:
                best_portfolio = copy.deepcopy(current_portfolio)
                best_quantities = current_quantities.copy()
                best_obj, best_budget = current_obj, current_budget
        
        if i % 200 == 0:
            print(f"  Iteration {i}, Best Objective: {best_obj:.4f}")
    
    portfolio = [[int(stock[COL_STOCK_ID]), stock[COL_SYMBOL], stock[COL_NAME],
                  stock[COL_PRICE], int(stock[COL_CATEGORY]), best_quantities[i]]
                 for i, stock in enumerate(best_portfolio)]
    
    time_taken = time.time() - start_time
    print(f"\n[Baseline HC] Completed in {time_taken:.2f} seconds")
    
    return portfolio, best_obj, best_budget, time_taken


#Main Running

def main():
    """Run all experiments and compare results."""
    print("\n" + "="*70)
    print("   GNN-ENHANCED STOCK PORTFOLIO OPTIMIZATION EXPERIMENTS")
    print("="*70)
    
    print("\nLoading dataset...")
    stocks_df = pd.read_csv('Final_Input_dataset_for_DSS.csv')
    print(f"Dataset loaded: {len(stocks_df)} stocks")
    
    # Experiment parameters
    PORTFOLIO_SIZE = 10
    LARGE_CAP_COUNT = 5
    MID_CAP_COUNT = 2
    TOTAL_BUDGET = 10000
    UPPER_BUDGET_LIMIT = 0.20
    LOWER_BUDGET_LIMIT = 0.05
    
    # Objective function weights (α1=0.5, α2=0.2, α3=0.25, α4=0.05)
    WEIGHT_FITNESS = 0.5
    WEIGHT_PERCENT_CHANGE = 0.2
    WEIGHT_REV_GROWTH = 0.25
    WEIGHT_NORMALIZED_BUDGET = 0.05
    
    SHLO_EPOCHS = 100
    SHLO_POP_SIZE = 50
    HC_ITERATIONS = 1000
    
    print(f"\nExperiment Parameters:")
    print(f"  - Portfolio Size: {PORTFOLIO_SIZE}")
    print(f"  - Large-cap: {LARGE_CAP_COUNT}, Mid-cap: {MID_CAP_COUNT}, Small-cap: {PORTFOLIO_SIZE - LARGE_CAP_COUNT - MID_CAP_COUNT}")
    print(f"  - Total Budget: ${TOTAL_BUDGET}")
    print(f"  - Budget Limits: {LOWER_BUDGET_LIMIT*100}% - {UPPER_BUDGET_LIMIT*100}%")
    print(f"  - SHLO: {SHLO_EPOCHS} epochs, population {SHLO_POP_SIZE}")
    print(f"  - Hill Climbing: {HC_ITERATIONS} iterations")
    
    results = {}
    
    portfolio, fitness, budget, time_taken = run_baseline_shlo(
        stocks_df, PORTFOLIO_SIZE, LARGE_CAP_COUNT, MID_CAP_COUNT,
        TOTAL_BUDGET, UPPER_BUDGET_LIMIT, LOWER_BUDGET_LIMIT,
        WEIGHT_FITNESS, WEIGHT_PERCENT_CHANGE, WEIGHT_REV_GROWTH, WEIGHT_NORMALIZED_BUDGET,
        epochs=SHLO_EPOCHS, pop_size=SHLO_POP_SIZE
    )
    results['Baseline SHLO'] = {
        'portfolio': portfolio,
        'objective': fitness,
        'budget_used': budget,
        'time': time_taken
    }
    
    portfolio, fitness, budget, time_taken, _ = run_gnn_enhanced_shlo(
        stocks_df, PORTFOLIO_SIZE, LARGE_CAP_COUNT, MID_CAP_COUNT,
        TOTAL_BUDGET, UPPER_BUDGET_LIMIT, LOWER_BUDGET_LIMIT,
        WEIGHT_FITNESS, WEIGHT_PERCENT_CHANGE, WEIGHT_REV_GROWTH, WEIGHT_NORMALIZED_BUDGET,
        epochs=SHLO_EPOCHS, pop_size=SHLO_POP_SIZE
    )
    results['GNN-SHLO'] = {
        'portfolio': portfolio,
        'objective': fitness,
        'budget_used': budget,
        'time': time_taken
    }
    
    portfolio, fitness, budget, time_taken = run_baseline_hill_climbing(
        stocks_df, PORTFOLIO_SIZE, LARGE_CAP_COUNT, MID_CAP_COUNT,
        TOTAL_BUDGET, UPPER_BUDGET_LIMIT, LOWER_BUDGET_LIMIT,
        WEIGHT_FITNESS, WEIGHT_PERCENT_CHANGE, WEIGHT_REV_GROWTH, WEIGHT_NORMALIZED_BUDGET,
        iterations=HC_ITERATIONS
    )
    results['Baseline HC'] = {
        'portfolio': portfolio,
        'objective': fitness,
        'budget_used': budget,
        'time': time_taken
    }
    
    portfolio, fitness, budget, time_taken, _ = run_gnn_enhanced_hill_climbing(
        stocks_df, PORTFOLIO_SIZE, LARGE_CAP_COUNT, MID_CAP_COUNT,
        TOTAL_BUDGET, UPPER_BUDGET_LIMIT, LOWER_BUDGET_LIMIT,
        WEIGHT_FITNESS, WEIGHT_PERCENT_CHANGE, WEIGHT_REV_GROWTH, WEIGHT_NORMALIZED_BUDGET,
        iterations=HC_ITERATIONS
    )
    results['GNN-HC'] = {
        'portfolio': portfolio,
        'objective': fitness,
        'budget_used': budget,
        'time': time_taken
    }
    
    generate_results_report(results, PORTFOLIO_SIZE, TOTAL_BUDGET, LARGE_CAP_COUNT, MID_CAP_COUNT)
    
    return results


def generate_results_report(results, portfolio_size, total_budget, large_cap, mid_cap):
    """Generate markdown report with results comparison."""
    
    report = f"""# GNN-Enhanced Stock Portfolio Optimization Results
    
    with open('GNN_EXPERIMENT_RESULTS.md', 'w') as f:
        f.write(report)
    
    print("\n" + "="*70)
    print("   RESULTS SAVED TO: GNN_EXPERIMENT_RESULTS.md")
    print("="*70)


if __name__ == "__main__":
    results = main()

