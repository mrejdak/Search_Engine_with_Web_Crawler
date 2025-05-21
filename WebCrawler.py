from collections import defaultdict, deque
import time
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import requests
import json
import scipy as sp


class WebCrawler:
    def __init__(self, start_urls, domain, skip_urls=[]):
        self.urls = deque(start_urls) if isinstance(start_urls, list) else deque([start_urls])
        self.domain = domain
        self.found_urls = set(start_urls) if isinstance(start_urls, list) else {
            start_urls}
        self.found_urls.update(skip_urls)
        self.stemmer = PorterStemmer()
        try:
            with open("documents.json", "r") as doc_file:
                self.documents = json.load(doc_file)
                self.found_urls.update(self.documents)
        except (json.JSONDecodeError, FileNotFoundError) as _:
            print("Failed to load documents.json")
            self.documents = []
        try:
            self.terms_by_doc = sp.sparse.load_npz("terms_by_doc.npz")
            self.terms_by_doc = self.terms_by_doc.tolil()
        except (EOFError, FileNotFoundError):
            print("Failed to load terms_by_doc.npz")
            self.terms_by_doc = sp.sparse.lil_matrix((0, 0))
        try:
            with open("terms.json", 'r') as t_file:
                self.terms = json.load(t_file)
        except (json.JSONDecodeError, FileNotFoundError):
            print("Failed to load terms.json")
            self.terms = {}
        try:
            with open("stop_words.txt", 'r') as sw_file:
                self.stop_words = set(sw_file.read().splitlines())
        except (json.JSONDecodeError, FileNotFoundError):
            print("Failed to load stop_words.txt")
            self.stop_words = set()

    def _crawl_page(self, url):
        page = requests.get(url)
        if page.status_code == 200:
            soup = BeautifulSoup(page.content, 'html.parser')
            links = soup.find_all("a")
            all_links = [link.get('href') for link in links]
            new_links = set([self.domain + link for link in all_links
                             if link is not None
                             and str(link).startswith("/wiki/")
                             and 'Special:' not in link
                             and ((':' not in link and '%3A' not in link) or 'Category:' in link)
                             and self.domain + link not in self.found_urls])
            self.urls.extend(new_links)
            self.found_urls.update(new_links)
            if 'Category:' in url:
                return []
            paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            return paragraphs
        else:
            print(f"Failed to retrieve: {url}\nStatus Code: {page.status_code}")
            return []

    def _create_index(self, paragraphs):
        bag_of_words = defaultdict(int)  # bag_of_words = {idx0 : cnt0, idx1 : cnt1, ...}
        for p in paragraphs:
            words = word_tokenize(p.text.lower())
            filtered_words = [self.stemmer.stem(word) for word in words if
                              word not in self.stop_words and word.isalpha()]
            for word in filtered_words:
                if word not in self.terms:
                    idx = len(self.terms)
                    self.terms[word] = idx
                bag_of_words[self.terms[word]] += 1
        return bag_of_words

    def _add_index_to_matrix(self, bag_of_words, url):
        self.terms_by_doc.resize((len(self.terms), self.terms_by_doc.shape[1] + 1))
        for idx, count in bag_of_words.items():
            self.terms_by_doc[idx, self.terms_by_doc.shape[1] - 1] = count
        self.documents.append(url)

    def _save_current(self):
        with open('terms.json', 'w') as t_file:
            json.dump(self.terms, t_file)
        with open('documents.json', 'w') as d_file:
            json.dump(self.documents, d_file)
        with open('current_urls.json', 'w') as url_file:
            json.dump(list(self.urls), url_file)
        csr_matrix = self.terms_by_doc.tocsr()
        sp.sparse.save_npz("terms_by_doc.npz", csr_matrix)

    def crawl(self, max_crawls=10, delay=1.):
        i = 0
        while len(self.urls) > 0 and i < max_crawls:
            url = self.urls.popleft()
            # if not url.startswith(self.domain):
            #     url = self.domain + url
            paragraphs = self._crawl_page(url)
            bag_of_words = self._create_index(paragraphs)
            if len(bag_of_words) > 0:
                self._add_index_to_matrix(bag_of_words, url)
            i += 1
            time.sleep(delay)
            if i % 1000 == 0:
                print("Saving batch...")
                self._save_current()
        self._save_current()


if __name__ == '__main__':
    try:
        with open('current_urls.json', 'r') as f:
            queued_urls = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        queued_urls = "https://simple.wikipedia.org/wiki/Category:Contents"

    wc = WebCrawler(start_urls=queued_urls, domain="https://simple.wikipedia.org",
                    skip_urls=["https://simple.wikipedia.org/wiki/Category:Noindexed_pages"])
    wc.crawl(max_crawls=50000, delay=0.75)