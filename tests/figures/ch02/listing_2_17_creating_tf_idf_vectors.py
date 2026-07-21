# Figure - Listing 2.17: Creating TF-IDF vectors
# Source: chapters/ch02.md lines 647-663
# Chapter: 2
# Category: needs-fixture  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
from sklearn.feature_extraction.text import TfidfVectorizer

movies["clean_title"] = (
    movies["title"]
    .str.replace(r"\s*\((?:19|20)\d{2}(?:-(?:19|20)\d{2})?\)\s*$", "", regex=True)
    .str.strip()
)  #A

movies['content'] = movies['clean_title'] + ' ' + movies['genres'].fillna('')  #B

vectorizer = TfidfVectorizer(
    max_features=500,
    stop_words='english',  #C
    token_pattern=r'(?u)\b\w+\b'
)

content_vectors = vectorizer.fit_transform(movies['content'])  #D

# Callout annotations (from the book):
#   #A remove years from titles, as they will otherwise be the things focused on.
#   #B Combine title and genres for richer representation
#   #C Remove common English words
#   #D Create TF-IDF vectors, limit to 500 features
