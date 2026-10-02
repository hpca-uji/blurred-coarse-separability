# Vendi imports
from vendi_score import vendi

# General external imports
import numpy as np

# Sklearn imports
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score

def compute_vendi(features):

    # Get embedding dimension from features
    embedding_dim = features[0].shape[1]

    # Concatenate all samples for the general Vendi score
    concat_features = np.concatenate(features, axis=0)

    # Compute general vendi
    score = vendi.score_dual(concat_features, normalize=True)

    # Compute normalized von neumman
    norm = np.log(score) / np.log(embedding_dim)
    print(f"General entropy: {score}; {norm}")


def compute_mi(features):

    # Get number of classes
    num_classes = len(features)
    
    # Number of samples in each class 
    num_samples = [class_data.shape[0] for class_data in features]

    # Create true labels
    y_true = np.concatenate([np.full(n, i, dtype=int) for i, n in enumerate(num_samples)])

    # Reshape
    concat_features = np.concatenate(features, axis=0)

    # Create clusters
    y_pred = KMeans(
        n_clusters=num_classes,
        n_init=1,
        random_state=0,
    ).fit_predict(concat_features)

    # Compute nmi
    nmi = normalized_mutual_info_score(y_true, y_pred)
    print(f"nmi: {nmi}")

    # Compute ami
    ami = adjusted_mutual_info_score(y_true, y_pred)
    print(f"ami: {ami}")