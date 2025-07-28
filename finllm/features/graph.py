import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import pickle
import os

class GraphFeatureProcessor:
    """
    Process graph-based features for FinLLM
    """
    def __init__(self):
        """Initialize the graph feature processor"""
        pass
    
    def build_stock_industry_graph(self, stock_info_df, industry_col='industry'):
        """
        Build a stock-industry graph from stock information
        
        Args:
            stock_info_df: DataFrame with stock information
            industry_col: Name of industry column
            
        Returns:
            NetworkX graph
        """
        G = nx.Graph()
        
        # Add stock nodes
        for idx, row in stock_info_df.iterrows():
            ticker = row['ticker']
            industry = row[industry_col]
            
            # Add stock node with attributes
            G.add_node(ticker, node_type='stock', industry=industry)
            
            # Add industry node if not exists
            if not G.has_node(industry):
                G.add_node(industry, node_type='industry')
            
            # Add edge between stock and industry
            G.add_edge(ticker, industry, weight=1.0)
        
        print(f"Built graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
        
        return G
    
    def add_stock_correlations(self, G, returns_df, min_corr=0.3, window=60):
        """
        Add stock correlation edges to graph
        
        Args:
            G: NetworkX graph
            returns_df: DataFrame with stock returns
            min_corr: Minimum correlation to add an edge
            window: Window size for rolling correlation
            
        Returns:
            Updated NetworkX graph
        """
        # Get stock nodes
        stock_nodes = [node for node, attr in G.nodes(data=True) if attr['node_type'] == 'stock']
        stock_tickers = [ticker for ticker in stock_nodes if ticker in returns_df.columns]
        
        print(f"Computing correlations for {len(stock_tickers)} stocks...")
        
        # Calculate rolling correlations for all pairs
        num_added = 0
        
        for i, ticker1 in enumerate(tqdm(stock_tickers)):
            for ticker2 in stock_tickers[i+1:]:
                # Skip if already connected
                if G.has_edge(ticker1, ticker2):
                    continue
                
                # Calculate correlation
                corr = returns_df[ticker1].rolling(window).corr(returns_df[ticker2])
                
                # Use mean correlation over the period
                mean_corr = corr.mean()
                
                # Add edge if correlation is strong enough
                if abs(mean_corr) >= min_corr:
                    G.add_edge(ticker1, ticker2, weight=abs(mean_corr), corr_sign=np.sign(mean_corr))
                    num_added += 1
        
        print(f"Added {num_added} correlation edges to graph")
        
        return G
    
    def add_news_connections(self, G, news_df, min_occurrences=2):
        """
        Add connections based on co-occurrence in news
        
        Args:
            G: NetworkX graph
            news_df: DataFrame with news articles
            min_occurrences: Minimum co-occurrences to add an edge
            
        Returns:
            Updated NetworkX graph
        """
        # Extract all tickers mentioned in each headline
        headline_tickers = []
        
        for _, row in tqdm(news_df.iterrows(), total=len(news_df), desc="Extracting ticker mentions"):
            headline = row['headline']
            ticker = row['ticker']
            
            # Find all tickers that appear in the headline
            mentioned_tickers = set()
            mentioned_tickers.add(ticker)  # Add the primary ticker
            
            # Get all stock nodes
            stock_nodes = [node for node, attr in G.nodes(data=True) if attr['node_type'] == 'stock']
            
            # Look for other tickers in the headline
            for other_ticker in stock_nodes:
                if other_ticker != ticker and other_ticker in headline:
                    mentioned_tickers.add(other_ticker)
            
            # Add to list if multiple tickers
            if len(mentioned_tickers) > 1:
                headline_tickers.append(list(mentioned_tickers))
        
        print(f"Found {len(headline_tickers)} headlines with multiple ticker mentions")
        
        # Count co-occurrences
        co_occurrences = {}
        
        for tickers in headline_tickers:
            for i, ticker1 in enumerate(tickers):
                for ticker2 in tickers[i+1:]:
                    pair = tuple(sorted([ticker1, ticker2]))
                    co_occurrences[pair] = co_occurrences.get(pair, 0) + 1
        
        # Add edges for co-occurrences
        num_added = 0
        
        for (ticker1, ticker2), count in co_occurrences.items():
            if count >= min_occurrences:
                # Skip if nodes don't exist in the graph
                if not G.has_node(ticker1) or not G.has_node(ticker2):
                    continue
                    
                # Update existing edge or create new one
                if G.has_edge(ticker1, ticker2):
                    G[ticker1][ticker2]['news_count'] = count
                else:
                    G.add_edge(ticker1, ticker2, weight=count/min_occurrences, news_count=count)
                    num_added += 1
        
        print(f"Added {num_added} news co-occurrence edges to graph")
        
        return G
    
    def convert_to_pytorch_geometric(self, G, node_features=None):
        """
        Convert NetworkX graph to PyTorch Geometric format
        
        Args:
            G: NetworkX graph
            node_features: Dictionary mapping node names to feature vectors
            
        Returns:
            PyTorch Geometric Data object
        """
        # Get unique node types
        node_types = set()
        for _, attr in G.nodes(data=True):
            node_types.add(attr.get('node_type', 'unknown'))
        
        # Create mapping from node names to indices
        node_to_idx = {node: i for i, node in enumerate(G.nodes())}
        
        # Create edge index
        edge_index = []
        edge_attr = []
        
        for u, v, data in G.edges(data=True):
            edge_index.append([node_to_idx[u], node_to_idx[v]])
            edge_index.append([node_to_idx[v], node_to_idx[u]])  # Add reverse edge for undirected graph
            
            # Get edge weight
            weight = data.get('weight', 1.0)
            edge_attr.append([weight])
            edge_attr.append([weight])  # Same weight for reverse edge
        
        # Convert to tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Create node feature matrix
        if node_features is not None:
            # Use provided features
            x = []
            for node in G.nodes():
                if node in node_features:
                    x.append(node_features[node])
                else:
                    # Use zeros for missing features
                    feature_dim = next(iter(node_features.values())).shape[0]
                    x.append(np.zeros(feature_dim))
        else:
            # Create simple one-hot encoding for node types
            node_type_to_idx = {t: i for i, t in enumerate(node_types)}
            
            x = []
            for node, attr in G.nodes(data=True):
                node_type = attr.get('node_type', 'unknown')
                one_hot = np.zeros(len(node_types))
                one_hot[node_type_to_idx[node_type]] = 1
                x.append(one_hot)
        
        # Convert to tensor
        x = torch.tensor(np.array(x), dtype=torch.float)
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        return data, node_to_idx
    
    def build_daily_graphs(self, returns_df, news_df, stock_info_df, window_size=30, step=1):
        """
        Build a time series of daily graphs
        
        Args:
            returns_df: DataFrame with daily returns
            news_df: DataFrame with news articles
            stock_info_df: DataFrame with stock information
            window_size: Size of rolling window for graph construction
            step: Step size for rolling window
            
        Returns:
            Dictionary mapping dates to graphs
        """
        daily_graphs = {}
        
        # Get unique dates from returns
        dates = returns_df.index.unique()
        
        for i in tqdm(range(window_size, len(dates), step), desc="Building daily graphs"):
            current_date = dates[i]
            window_start_date = dates[i - window_size]
            
            # Get window data
            window_returns = returns_df.loc[window_start_date:current_date]
            window_news = news_df[(news_df['date'] >= window_start_date) & 
                                 (news_df['date'] <= current_date)]
            
            # Build base graph
            G = self.build_stock_industry_graph(stock_info_df)
            
            # Add correlations
            G = self.add_stock_correlations(G, window_returns)
            
            # Add news connections
            G = self.add_news_connections(G, window_news)
            
            # Store graph
            daily_graphs[current_date] = G
        
        return daily_graphs
    
    def extract_graph_features(self, graph, ticker):
        """
        Extract graph features for a specific ticker
        
        Args:
            graph: NetworkX graph
            ticker: Stock ticker
            
        Returns:
            Dictionary of graph features
        """
        features = {}
        
        # Check if ticker exists in the graph
        if not graph.has_node(ticker):
            return features
        
        # Basic features
        features['degree'] = graph.degree(ticker)
        features['weighted_degree'] = sum(w.get('weight', 1.0) for _, _, w in graph.edges(ticker, data=True))
        
        # Centrality measures
        centrality_measures = {
            'betweenness_centrality': nx.betweenness_centrality(graph, k=10),
            'closeness_centrality': nx.closeness_centrality(graph),
            'eigenvector_centrality': nx.eigenvector_centrality(graph, max_iter=300)
        }
        
        for measure_name, measure_dict in centrality_measures.items():
            features[measure_name] = measure_dict.get(ticker, 0)
        
        # Local clustering coefficient
        features['clustering'] = nx.clustering(graph, ticker)
        
        # Number of connected stocks in the same industry
        industry = graph.nodes[ticker].get('industry')
        same_industry_neighbors = [n for n in graph.neighbors(ticker) if 
                                  graph.nodes[n].get('node_type') == 'stock' and
                                  graph.nodes[n].get('industry') == industry]
        features['same_industry_connections'] = len(same_industry_neighbors)
        
        # Number of news-based connections
        news_connections = sum(1 for _, _, data in graph.edges(ticker, data=True) 
                              if 'news_count' in data)
        features['news_connections'] = news_connections
        
        # Strong correlations (above 0.5)
        strong_corr = sum(1 for _, _, data in graph.edges(ticker, data=True) 
                         if data.get('weight', 0) > 0.5)
        features['strong_correlations'] = strong_corr
        
        return features
    
    def extract_daily_graph_features(self, daily_graphs, tickers):
        """
        Extract daily graph features for a list of tickers
        
        Args:
            daily_graphs: Dictionary mapping dates to graphs
            tickers: List of stock tickers
            
        Returns:
            DataFrame of daily graph features for each ticker
        """
        all_features = {}
        
        for date, graph in tqdm(daily_graphs.items(), desc="Extracting graph features"):
            date_features = {}
            
            for ticker in tickers:
                # Extract features for this ticker
                ticker_features = self.extract_graph_features(graph, ticker)
                
                # Add to date features with ticker prefix
                for feat_name, feat_value in ticker_features.items():
                    date_features[f'{ticker}_{feat_name}'] = feat_value
            
            all_features[date] = date_features
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(all_features, orient='index')
        
        return df