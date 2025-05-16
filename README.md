# Search Engine with Web Crawler

A simple search engine implementation using a custom web crawler and information retrieval techniques such as TF-IDF and Latent Semantic Analysis (SVD).
User interface created with ```gradio``` framework.

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


## SVD Parameter Comparison

The performance and relevance of results can vary with the number of components `k` used in Truncated SVD. Below is a comparison of search results with different `k` values.

```query```: `computer science`

| k=0 | k=50 | k=100 | k=200 |
|-----|------|-------|-------|
|     |      |       |       |
|     |      |       |       |
|     |      |       |       |
|     |      |       |       |
|     |      |       |       |

