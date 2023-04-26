import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Calculate cosine similarity matrix
from sklearn.metrics.pairwise import cosine_similarity

# Implement custom PCA with cosine similarity
class CosinePCA(PCA):
    def __init__(self, n_components=None):
        super(CosinePCA, self).__init__(n_components)
    
    def fit(self, X):
        # Centering the data
        X_centered = X - np.mean(X, axis=0)
        
        # Calculate cosine similarity matrix
        cosine_sim = cosine_similarity(X_centered)
        
        # Perform eigendecomposition on the cosine similarity matrix
        eigvals, eigvecs = np.linalg.eig(cosine_sim)
        
        # Sort eigenvectors and eigenvalues in descending order
        sort_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sort_indices]
        eigvecs = eigvecs[:, sort_indices]
        
        # Extract the top k eigenvectors
        if self.n_components is not None:
            eigvecs = eigvecs[:, :self.n_components]
        
        # Set the eigenvectors as the components
        self.components_ = eigvecs.T
        self.explained_variance_ = eigvals
        self.mean_ = np.mean(X, axis=0)
        
        return self

def cosine_pca_2d(X):
    pca = CosinePCA(n_components=2)
    data_pca = pca.fit_transform(X)
    return data_pca

def scatter_pair(data_1, iter_1, data_2, iter_2, labels):
    # Create a figure with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Create scatterplots on each subplot
    ax1.scatter(data_1[:, 0], data_1[:, 1], c=labels, label='Scatterplot 1')
    ax2.scatter(data_2[:, 0], data_2[:, 1], c=labels, label='Scatterplot 2')

    # Set titles and labels for each subplot
    ax1.set_title(f'Iteration: {iter_1}')
    ax1.set_xlabel('PCA1')
    ax1.set_ylabel('PCA2')
    ax2.set_title(f'Iteration: {iter_2}')
    ax2.set_xlabel('PCA1')
    ax2.set_ylabel('PCA2')

    # Add legend to each subplot
    # ax1.legend()
    # ax2.legend()

    return plt
    
def plot_model_records(data, baseline=None):
    # Creating subplots for each metric
    metrics = ['polarity_score', 'similarity_score', 'overfitting_indicator']
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow'][:len(data.items())]
    colors = colors * 3

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    metric_labels = ["Polarity Score", "Semantic Similarity Score"]

    for i, metric in enumerate(metrics[:2]):
        if metric in ['polarity_score', 'similarity_score'] and baseline:
            axs[i].axhline(y=baseline[metric], linestyle='dotted', color='gray', label='Reference')
        for model, model_data in data.items():
            epochs = [d['epoch'] for d in model_data]
            values = [d[metric] for d in model_data]
            axs[i].plot(epochs, values, label=model, color=colors.pop(0))
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(metric_labels[i])
        axs[i].set_title(metric_labels[i])
        # axs[i].set_ylabel(metric)
        # axs[i].set_title(metric)
        axs[i].legend()

    return plt