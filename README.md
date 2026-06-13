# Unsupervised Topic Discovery on BBC News Articles

K-Means clustering applied to BBC news text data to discover latent topics without using labels.

Two approaches for text vectorization:

- **Doc2Vec** (PV-DM) -- learns paragraph vectors directly from the corpus
- **Word2Vec + Average Pooling** -- averages word vectors per document to get a fixed-size representation

The notebook walks through the whole pipeline: loading data from Kaggle, cleaning and lemmatizing text, building vector representations, running K-Means for different values of K, and evaluating cluster quality with silhouette scores and cosine similarity inspection.

---

## Contents

<details>
<summary><b>notebook.ipynb</b></summary>

K-Means pipeline using Word2Vec + average pooling. Covers:

- Loading the BBC full-text dataset via `kagglehub`
- Text cleaning, stopword removal, lemmatization
- Word2Vec training + document vector generation via word-vector averaging
- K-Means clustering across K values
- Silhouette score evaluation
- Manual cluster inspection via cosine similarity against centroids
</details>

<details>
<summary><b>k_means_news_topic_modeling_v2.ipynb</b></summary>

Extended version using Doc2Vec (PV-DM) instead of averaged Word2Vec. Same preprocessing + evaluation steps, but with paragraph vectors learned directly from the document collection.
</details>

<details>
<summary><b>Flowcharts</b></summary>

There are two flowcharts in the repo showing the architecture of each approach:

- `doc2vec.png` -- how the Doc2Vec model feeds into K-Means
- `word2vec&avgpooling.png` -- how Word2Vec averaged embeddings are used for clustering
</details>

---

## Requirements

- Python 3.8+
- See `requirements.txt` for package versions

---

## How to run

Open `notebook.ipynb` in Google Colab or Jupyter and run cells top to bottom. The notebook downloads the BBC dataset automatically via `kagglehub` -- no manual data setup needed.

---

## Dataset

[BBC Full Text Document Classification](https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification) from Kaggle. 2225 news articles across 5 categories (business, entertainment, politics, sport, tech). The labels are only used here for post-hoc evaluation -- the clustering itself is fully unsupervised.

---

## Project structure

```
bbc-topic-discovery/
  notebook.ipynb              -- main notebook (Word2Vec + AvgPooling + K-Means)
  k_means_news_topic_modeling_v2.ipynb  -- variant using Doc2Vec
  doc2vec.png                 -- flowchart for Doc2Vec pipeline
  word2vec&avgpooling.png     -- flowchart for Word2Vec pipeline
  requirements.txt            -- dependencies
  README.md                   -- this file
```
