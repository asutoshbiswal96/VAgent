import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict


class SimpleRAG:
    """A tiny RAG using TF-IDF for retrieval over per-policyholder notes.

    Each policyholder row is a doc consisting of `notes`, `due_date`, `premium_amount`, and redacted metadata.
    """

    def __init__(self):
        self.df = None
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.nn = None
        self.embeddings = None

    def index_from_csv(self, path: str):
        self.df = pd.read_csv(path, dtype=str).fillna("")
        docs = self._docs_from_df(self.df)
        self.embeddings = self.vectorizer.fit_transform(docs)
        self.nn = NearestNeighbors(n_neighbors=3, metric='cosine').fit(self.embeddings)

    def _docs_from_df(self, df: pd.DataFrame) -> List[str]:
        docs = []
        for _, r in df.iterrows():
            parts = [f"PolicyID: {r['policy_id']}"]
            for k in ['name', 'due_date', 'premium_amount', 'notes']:
                if k in r and r[k]:
                    parts.append(f"{k}: {r[k]}")
            docs.append('\n'.join(parts))
        return docs

    def retrieve(self, policy_id: str, k: int = 2) -> List[Dict[str, str]]:
        """Return top-k docs (as dict rows) for the given policy_id. If policy_id exists, return its own row first."""
        if self.df is None:
            raise RuntimeError("Index not built")
        # find exact row
        row = self.df[self.df['policy_id'] == policy_id]
        if not row.empty:
            return [row.iloc[0].to_dict()]
        # otherwise retrieve by nearest neighbors on policy_id text
        query = f"policy_id: {policy_id}"
        qv = self.vectorizer.transform([query])
        dists, idxs = self.nn.kneighbors(qv, n_neighbors=min(k, self.embeddings.shape[0]))
        results = []
        for i in idxs[0]:
            results.append(self.df.iloc[i].to_dict())
        return results
