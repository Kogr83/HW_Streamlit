import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from eda import create_dataset, save_dataset, open_dataset, stat_quant, stat_cat, graphs_quant, correlation, graphs_causality, mis, cluster_analysis, graph_clusters, profile_clusters

df = create_dataset()


