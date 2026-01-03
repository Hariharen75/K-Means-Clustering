import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
import joblib

st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")
st.title("🛒 Mall Customer Segmentation with K-Means")

# Sidebar for file upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Data loaded successfully!")
    
    # Data preview
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
    with col2:
        st.subheader("Dataset Info")
        st.dataframe(pd.DataFrame({"Non-Null Count": df.count(), "Dtype": df.dtypes}))
    
    # Select features for clustering
    st.subheader("🎯 Feature Selection")
    features = st.multiselect("Select features for clustering (default: Income & Spending)", 
                             df.columns, 
                             default=['Annual Income (k$)', 'Spending Score (1-100)'])
    
    if len(features) >= 2:
        X = df[features]
        
        # Elbow plot
        st.subheader("📈 Elbow Method - Find Optimal K")
        max_k = st.slider("Max K for elbow plot", 1, 12, 10)
        sse = []
        k_range = range(1, max_k + 1)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            sse.append(kmeans.inertia_)
        
        fig_elbow = px.line(x=list(k_range), y=sse, 
                           labels={'x': 'K', 'y': 'SSE'}, 
                           title="Elbow Plot")
        st.plotly_chart(fig_elbow, use_container_width=True)
        
        # Optimal K selection
        optimal_k = st.slider("Select optimal K (recommended ~5)", 2, 10, 5)
        
        # Scaling and clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        df['Cluster'] = clusters
        
        # Cluster visualization
        st.subheader("🔍 Clusters Visualization")
        fig_clusters = px.scatter(df, x=features[0], y=features[1], 
                                color='Cluster', size_max=10,
                                title=f"Customer Segments (K={optimal_k})",
                                hover_data=df.columns)
        st.plotly_chart(fig_clusters, use_container_width=True)
        
        # FIXED: Centroids on original scale
        st.subheader("⭐ Centroids (Original Scale)")
        centers_scaled = kmeans.cluster_centers_
        centers_original = scaler.inverse_transform(centers_scaled)
        centers_df = pd.DataFrame(centers_original, columns=features)
        centers_df['Cluster'] = range(optimal_k)
        
        fig_centroids = px.scatter(centers_df, x=features[0], y=features[1], 
                                 color='Cluster', size_max=15,
                                 title=f"Centroids (K={optimal_k})")
        fig_centroids.update_traces(marker=dict(symbol='star', size=15, 
                                              line=dict(width=2, color='black')))
        st.plotly_chart(fig_centroids, use_container_width=True)
        
        # Cluster summary
        st.subheader("📊 Cluster Summary")
        cluster_summary = df.groupby('Cluster')[features].mean().round(2)
        st.dataframe(cluster_summary)
        
        # Download processed data
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button("💾 Download Clustered Data", csv_buffer.getvalue(), 
                          "clustered_customers.csv", "text/csv")
        
        # Save model (optional)
        if st.button("💾 Save Scaler & Model"):
            joblib.dump(scaler, "scaler.pkl")
            joblib.dump(kmeans, "kmeans.pkl")
            st.success("✅ Models saved as scaler.pkl & kmeans.pkl!")
    
    else:
        st.warning("⚠️ Please select at least 2 features for clustering.")
        
else:
    st.info("👈 Upload Mall_Customers.csv to get started!")
    st.markdown("**Expected columns:** CustomerID, Genre, Age, 'Annual Income (k$)', 'Spending Score (1-100)'")

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ for customer segmentation*")