# Figure — Listing 2.17: Creating TF-IDF vectors
# Source: chapters/ch02.md lines 647-663
# Chapter: 2
# Category: needs-fixture  (executable=True, expected=pass)
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A remove years from titles, as they will otherwise be the things focused on.
#   #B Combine title and genres for richer representation
#   #C Remove common English words
#   #D Create TF-IDF vectors, limit to 500 features
from sklearn.feature_extraction.text import TfidfVectorizer

movies["clean_title"] = (
    movies["title"]
    .str.replace(r"\s*\((?:19|20)\d{2}(?:-(?:19|20)\d{2})?\)\s*$", "", regex=True)
    .str.strip()
)

movies['content'] = movies['clean_title'] + ' ' + movies['genres'].fillna('')

vectorizer = TfidfVectorizer(
    max_features=500,
    stop_words='english',
    token_pattern=r'(?u)\b\w+\b'
)

content_vectors = vectorizer.fit_transform(movies['content'])
