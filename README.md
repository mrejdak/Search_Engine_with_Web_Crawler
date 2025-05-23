# Search Engine with Web Crawler

A simple search engine implementation using a custom web crawler and information retrieval techniques such as TF-IDF, SVD and low rank approximation.
User interface created with ```gradio``` framework.

![UI](img/search_engine_default.png)

## Getting Started

### Installing required packages

```bash
pip install -r requirements.txt
````

### Running the Project

**To run WebCrawler:**

```bash
python WebCrawler.py
```

_Files [documents.json](documents.json), [terms](terms.json), [terms_by_doc.npz](terms_by_doc.npz) already contain indexed data from 10000 wikipedia pages._

_**Update:** WebCrawler was slightly modified to allow continuation of crawling after rerunning the program. However, due to those changes, it is required to delete those files before running it (otherwise it stops crawling immediately)_

**To run the search engine:**

```bash
python main.py
```
and click on the local URL.

![example](img/search_engine_example.png)


## Implementation details

### Collecting and indexing data
This is accomplished by the custom WebCrawled tailored for crawling the Simplified English Wikipedia. It sends HTTP request using `request` library and parses the page with `BeautifulSoup`, extracting all text and links leading to other wikipedia pages, excluding non-content pages.

Extracted text is then tokenized and stemmed using `word_tokenize` and `PorterStemmer` tools from `nltk` library, as well as filtered for any [stop words](stop_words.txt) or non alphabet tokens. Then it is converted into a sparse **bag_of_words** vector and added to a **terms_by_doc** matrix.

All encountered words are stored in the **terms** dictionary, which contains their corresponding indexes in **bag_of_words** vectors.

### Preprocessing the *term_by_doc* matrix
First, TF-IDF and L2 norm normalization is done using `TfidfTransformer` and `normalize` from `scikit-learn` library. 
Then, if value of `k` passed during search query is > 0, a noise reduction is performed via SVD (Singular Value Decomposition) and low rank approximation (using `TruncatedSVD` from `scikit-learn`).
After reducing the dimensions of a matrix, a hnsw-index is created using HNSW algorithm (Hierarchical Navigable Small World) from `hnswlib` library, allowing for performing quick ANN (Approximate Nearest Neighbor) search.
Both the reduced matrix and the ANN index are then saved for future use.

### Querying
Inputted query is preprocessed the same way as crawled pages, resulting in a **bag_of_words** vector. Then depending on provided value of `k`, an original or reduced matrix is used and search is performed using either cosine similarity or the ANN index, returning 20 best fitted search results.

## SVD Parameter Comparison

The performance and relevance of results can vary with the number of components `k` used in Truncated SVD. Below is a comparison of search results with different `k` values.

These results were acquired from 100000 documents containing 222748 different words.
Queries for k >= 250 were carried out using ANN index and the rest - by calculating cosine similarity.

```query```: `bloody war in the middle ages`

| k=0                           | k=50                                 | k=100                        | k=250                        | k=350                        | k=500                          |
|-------------------------------|--------------------------------------|------------------------------|------------------------------|------------------------------|--------------------------------|
| High_Middle_Ages (0.280)      | Joseph_E._Johnston (0.948)           | Yugoslav_Wars (0.875)        | Nigerian_Civil_War (0.732)   | Limited_war (0.694)          | Limited_war (0.648)            |
| Early_medieval_period (0.238) | Battle_of_Gökçay_(1511) (0.942)      | Limited_war (0.854)          | Limited_war (0.731)          | Nigerian_Civil_War (0.677)   | Armed_conflict (0.623)         |
| Sino-Japanese_War (0.203)     | Battle_of_Fair_Oaks (0.973)          | Nigerian_Civil_War (0.848)   | Mozambican_Civil_War (0.712) | Mozambican_Civil_War (0.668) | High_Middle_Ages (0.618)       |
| Late_middle_ages (0.200)      | Romanian_War_of_Independence (0.934) | Theater_(warfare) (0.842)    | Inca_Civil_War (0.705)       | War (0.659)                  | Sino-Japanese_War (0.590)      |
| Middle_Ages (0.200)           | Russo-Turkish_war (0.933)            | Mozambican_Civil_War (0.841) | Sino-Japanese_War (0.701)    | Sino-Japanese_War (0.649)    | Stop_the_War_Coalition (0.584) |

As we can see, search without SVD mainly focuses on middle ages as the time period. SVD with lower k values remove a lot of information leading to results concerning seemingly uncorrelated wars and battles, while SVD with higher values return results more focused on meaning of 'war'.
What's interesting is that the only for k=50 the search found a war/battle that happened in middle ages (Battle of Gökçay).

---
```query```: `computer science`

| k=0                                  | k=50                                 | k=100                                | k=250                                       | k=350                                       | k = 500                                     |
|--------------------------------------|--------------------------------------|--------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|
| Theoretical_computer_science (0.545) | Theoretical_computer_science (0.971) | Theoretical_computer_science (0.987) | Theoretical_computer_science (0.976)        | Theoretical_computer_science (0.962)        | Theoretical_computer_science (0.954)        |
| Information_science (0.491)          | Systems_science (0.899               | Computer_science (0.893)             | Clarence_Ellis_(computer_scientist) (0.894) | Clarence_Ellis_(computer_scientist) (0.886) | Computer_science (0.873)                    |
| Computer_science (0.491)             | Computer_engineering (0.888)         | Information_science (0.893)          | Computer_science (0.893)                    | Computer_science (0.877)                    | Computing (0.873)                           |
| Natural_sciences (0.428)             | Manufacturing_engineering (0.875)    | Computing (0.893)                    | Information_science (0.893)                 | Computing (0.877)                           | Information_science (0.873)                 |
| Computation (0.425)                  | Search (0.851)                       | Computable_function (0.892)          | Computing (0.893)                           | Information_science (0.877)                 | Clarence_Ellis_(computer_scientist) (0.865) |

This time, when provided with much simpler query, all tested values of k give quite similar results.

## Conclusions

The number of components used in SVD (k) significantly influences the relevance and specificity of search results of more advanced queries.
Lower values (e.g., k=50) tend to generalize those queries losing a lot of information, often returning loosely related topics.
For simpler queries, SVD does not have much influence on the results.

Approximate Nearest Neighbors (ANN) search method did not provide any significant time improvements, for higher values of `k` (like 500) it even made the process longer.
This might be improved by changing hyperparameters of the created hnsw-index.

Search Engine seems to prioritize shorter articles, probably because of the use of TF-IDF.