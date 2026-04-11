import argparse
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

# -----------------------------
# Argument Parser
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--categories', nargs='+', default=None)
parser.add_argument('--components', nargs='+', type=int, required=True)

args = parser.parse_args()

# -----------------------------
# Load Dataset
# -----------------------------
dataset = fetch_20newsgroups(
    subset='all',
    categories=args.categories,
    remove=('headers', 'footers', 'quotes')
)

documents = dataset.data
labels = dataset.target

print(f"Loaded {len(documents)} documents")

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    max_df=0.5,
    min_df=2,
    stop_words='english'
)

X_tfidf = vectorizer.fit_transform(documents)

terms = vectorizer.get_feature_names_out()

# -----------------------------
# Output files
# -----------------------------
results = []

topic_file = open("lsa_topic_terms.txt", "w")

# -----------------------------
# Loop over components
# -----------------------------
for n_comp in args.components:
    print(f"\nRunning SVD with {n_comp} components")

    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X_reduced = lsa.fit_transform(X_tfidf)

    explained_variance = svd.explained_variance_ratio_.sum()

    # -----------------------------
    # Clustering (KMeans)
    # -----------------------------
    true_k = len(set(labels))

    km = KMeans(n_clusters=true_k, random_state=42)
    cluster_labels = km.fit_predict(X_reduced)

    silhouette = silhouette_score(X_reduced, cluster_labels)

    print(f"Explained Variance: {explained_variance:.4f}")
    print(f"Silhouette Score: {silhouette:.4f}")

    results.append({
        "components": n_comp,
        "explained_variance": explained_variance,
        "silhouette_score": silhouette
    })

    # -----------------------------
    # Topic Interpretation
    # -----------------------------
    topic_file.write(f"\n=== Topics for {n_comp} components ===\n")

    for i, comp in enumerate(svd.components_[:5]):  # top 5 topics
        terms_idx = np.argsort(comp)[::-1][:10]
        topic_terms = [terms[idx] for idx in terms_idx]

        topic_file.write(f"Topic {i+1}: {', '.join(topic_terms)}\n")

topic_file.close()

# -----------------------------
# Save Results CSV
# -----------------------------
df = pd.DataFrame(results)
df.to_csv("svd_results.csv", index=False)

print("\nSaved results to svd_results.csv and lsa_topic_terms.txt")