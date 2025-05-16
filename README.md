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
_Files [documents.json](documents.json),[terms](terms.json), [terms_by_doc.npz](terms_by_doc.npz) already contain indexed data from 10000 wikipedia pages._

To run the search engine:

```bash
python main.py
```
and click on the local URL.

![example](img/search_engine_example.png)

## SVD Parameter Comparison

The performance and relevance of results can vary with the number of components `k` used in Truncated SVD. Below is a comparison of search results with different `k` values.

```query```: `bloody war in middle ages`

| k=0                       | k=50                      | k=100                     | k=250                     |
|---------------------------|---------------------------|---------------------------|---------------------------|
| High_Middle_Ages (0.415)  | Early_Middle_Ages (0.817) | High_Middle_Ages (0.752)  | High_Middle_Ages (0.884)  |
| Early_Middle_Ages (0.345) | High_Middle_Ages (0.814)  | Early_Middle_Ages (0.743) | Early_Middle_Ages (0.739) |
| Middle_Ages (0.287)       | Modern_History (0.779)    | Medieval_warfare (0.689)  | Medieval (0.627)          |
| Medieval (0.287)          | Licinius (0.760)          | Medieval (0.689)          | Medieval_warfare (0.627)  |
| Medieval_warfare (0.287)  | Medieval_warfare (0.756)  | Middle_Ages (0.689)       | Middle_Ages (0.627)       |

---
```query```: `computer science`

| k=0                                  | k=50                                 | k=100                                | k=250                                |
|--------------------------------------|--------------------------------------|--------------------------------------|--------------------------------------|
| Theoretical_computer_science (0.545) | Theoretical_computer_science (0.982) | Theoretical_computer_science (0.963) | Theoretical_computer_science (0.933) |
| Natural_sciences (0.432)             | Computing (0.951)                    | Computer_science (0.918)             | Computing (0.829)                    |
| Computer_science (0.426)             | Computer_science (0.951)             | Computing (0.918)                    | Computer_science (0.829)             |
| Computing (0.462)                    | Computability_theory (0.944)         | Computability_theory (0.880)         | Computation (0.737)                  |
| Computers (0.407)                    | Debugging (0.935)                    | Distributed_computing (0.863)        | Computer_vision (0.719)              |

