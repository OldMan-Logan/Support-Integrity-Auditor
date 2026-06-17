import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# Load model once
# --------------------------------------------------

MODEL_NAME = "BAAI/bge-small-en-v1.5"

model = SentenceTransformer(MODEL_NAME)

# --------------------------------------------------
# Reference severity prototypes
# --------------------------------------------------

LOW_EXAMPLES = [
    "Forgot password",
    "Need to update profile picture",
    "Unable to change username",
    "Minor UI issue",
    "Typo in dashboard"
]

MEDIUM_EXAMPLES = [
    "Cannot login to account",
    "Order delayed",
    "Invoice missing",
    "Email notifications not working"
]

HIGH_EXAMPLES = [
    "Website unavailable for many users",
    "Payment failures affecting customers",
    "Application crashes repeatedly",
    "Database latency causing outages"
]

CRITICAL_EXAMPLES = [
    "Production server down",
    "Security breach detected",
    "Complete payment gateway outage",
    "All customers blocked from accessing service",
    "Massive data loss"
]

# --------------------------------------------------
# Precompute prototype embeddings
# --------------------------------------------------

low_emb = model.encode(
    LOW_EXAMPLES,
    normalize_embeddings=True
)

medium_emb = model.encode(
    MEDIUM_EXAMPLES,
    normalize_embeddings=True
)

high_emb = model.encode(
    HIGH_EXAMPLES,
    normalize_embeddings=True
)

critical_emb = model.encode(
    CRITICAL_EXAMPLES,
    normalize_embeddings=True
)

# --------------------------------------------------
# Utility
# --------------------------------------------------

def average_similarity(ticket_embedding, prototype_embeddings):
    """
    Returns average cosine similarity to a prototype group.
    """

    sims = cosine_similarity(
        ticket_embedding.reshape(1, -1),
        prototype_embeddings
    )[0]

    return float(np.mean(sims))


# --------------------------------------------------
# Main severity estimator
# --------------------------------------------------

def get_embedding_severity(text):
    """
    Returns:

    {
        "severity": "...",
        "score": float,
        "similarities": {...}
    }
    """

    emb = model.encode(
        text,
        normalize_embeddings=True
    )

    low_sim = average_similarity(emb, low_emb)
    med_sim = average_similarity(emb, medium_emb)
    high_sim = average_similarity(emb, high_emb)
    critical_sim = average_similarity(emb, critical_emb)

    sims = {
        "Low": low_sim,
        "Medium": med_sim,
        "High": high_sim,
        "Critical": critical_sim
    }

    severity = max(sims, key=sims.get)

    score_map = {
        "Low": 0.0,
        "Medium": 0.33,
        "High": 0.66,
        "Critical": 1.0
    }

    return {
        "severity": severity,
        "score": score_map[severity],
        "similarities": sims
    }


# --------------------------------------------------
# Batch inference
# --------------------------------------------------

def add_embedding_scores(df, text_column):
    """
    Adds:

    embedding_severity
    embedding_score
    """

    severities = []
    scores = []

    for text in df[text_column]:

        result = get_embedding_severity(text)

        severities.append(result["severity"])
        scores.append(result["score"])

    df["embedding_severity"] = severities
    df["embedding_score"] = scores

    return df