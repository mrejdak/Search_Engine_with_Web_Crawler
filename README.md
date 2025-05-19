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

Web crawler is located (for now) in the jupyter notebook WebCrawler.ipynb. To run it, execute all cells in the notebook, following the instructions in it.  
_Files [documents.json](documents.json), [terms](terms.json), [terms_by_doc.npz](terms_by_doc.npz) already contain indexed data from 10000 wikipedia pages._

To run the search engine:

```bash
python main.py
```
and click on the local URL.

![example](img/search_engine_example.png)

## SVD Parameter Comparison

The performance and relevance of results can vary with the number of components `k` used in Truncated SVD. Below is a comparison of search results with different `k` values.

These results were acquired from 10000 documents containing 51435 different words.
Queries for k equal 250 and 500 were carried out using ANN and the rest - using cosine similarity.

```query```: `bloody war in middle ages`

| k=0                       | k=50                      | k=100                        | k=250                        | k=500                     |
|---------------------------|---------------------------|------------------------------|------------------------------|---------------------------|
| High_Middle_Ages (0.279)  | Peter_I_of_Russia (0.885) | World_War_I (0.794)          | High_Middle_Ages (0.686)     | High_Middle_Ages (0.730)  |
| Early_Middle_Ages (0.232) | World_War_I (0.877)       | World_War_II (0.758)         | Early_Middle_Ages (0.604)    | Early_Middle_Ages (0.625) |
| Middle_Ages (0.203)       | Nazi_Germany (0.857)      | Nazi_Germany (0.753)         | Warfare (0.603)              | Middle_Ages (0.570)       |
| Medieval (0.203)          | World_War_II (0.838)      | Peter_I_of_Russia (0.748)    | Conventional_warfare (0.571) | Medieval_warfare (0.570)  |
| Medieval_warfare (0.203)  | Anglo-Dutch_War (0.836)   | Conventional_warfare (0.748) | Middle_Ages (0.545)          | Medieval (0.570)          |

As we can see, lower k values remove a lot of information leading to more general results concerning war, while higher values return results focused on middle ages. Another thing to consider is rather small batch of documents this query was performed on, which might be the reason for lower accuracy of results.

---
```query```: `computer science`

| k=0                                  | k=50                                 | k=100                                | k=250                                | k = 500                              |
|--------------------------------------|--------------------------------------|--------------------------------------|--------------------------------------|--------------------------------------|
| Theoretical_computer_science (0.545) | Theoretical_computer_science (0.982) | Theoretical_computer_science (0.963) | Theoretical_computer_science (0.933) | Theoretical_computer_science (0.920) |
| Natural_sciences (0.432)             | Computing (0.951)                    | Computer_science (0.918)             | Computing (0.829)                    | Computer_science (0.703)             |
| Computer_science (0.426)             | Computer_science (0.951)             | Computing (0.918)                    | Computer_science (0.829)             | Computing (0.703)                    |
| Computing (0.426)                    | Computability_theory (0.944)         | Computability_theory (0.880)         | Computation (0.737)                  | Computation (0.690)                  |
| Computers (0.407)                    | Debugging (0.935)                    | Distributed_computing (0.863)        | Computer_vision (0.719)              | Computer_vision (0.661)              |

This time, when provided with much simpler query, all tested values of k give quite similar results.

## Conclusions

The number of components used in SVD (k) significantly influences the relevance and specificity of search results of more advanced queries.
Lower values (e.g., k=50) tend to generalize those queries, often returning loosely related topics.
For simpler queries, SVD does not have much influence on the results.

Approximate Nearest Neighbors (ANN) search method did not provide any significant time improvements, however this is probably due to the small dataset that those tests were concluded on.