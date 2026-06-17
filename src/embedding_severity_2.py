from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

def add_embedding_scores(df, text_column):

    model = SentenceTransformer(
        "BAAI/bge-small-en-v1.5"
    )

    embeddings = model.encode(
        df[text_column].tolist(),
        normalize_embeddings=True
    )


    kmeans = KMeans(
        n_clusters=4,
        random_state=42
    )

    clusters = kmeans.fit_predict(
        embeddings
    ) 

    df["cluster"] = clusters

    return df