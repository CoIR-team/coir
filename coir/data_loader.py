import json
import logging
from io import StringIO
from typing import Dict, Tuple
import csv
from datasets import load_dataset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

class InMemoryDataLoader:
    def __init__(self, corpus_data, query_data, qrels_data):
        self.corpus_file = StringIO('\n'.join(json.dumps(doc) for doc in corpus_data))
        self.query_file = StringIO('\n'.join(json.dumps(query) for query in query_data))
        self.qrels_file = StringIO('\n'.join(f"{qrel['query_id']}\t{qrel['corpus_id']}\t{qrel['score']}" for qrel in qrels_data))
        self.corpus = {}
        self.queries = {}
        self.qrels = {}

    def load_custom(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
        logger.info("Loading Corpus...")
        self._load_corpus()
        logger.info("Loaded %d Documents.", len(self.corpus))
        logger.info("Doc Example: %s", list(self.corpus.values())[0])

        logger.info("Loading Queries...")
        self._load_queries()

        self._load_qrels()
        self.queries = {qid: self.queries[qid] for qid in self.qrels}
        logger.info("Loaded %d Queries.", len(self.queries))
        logger.info("Query Example: %s", list(self.queries.values())[0])

        return self.corpus, self.queries, self.qrels

    def _load_corpus(self):
        self.corpus_file.seek(0)  # Reset the StringIO object to the beginning
        for line in tqdm(self.corpus_file):
            doc = json.loads(line)
            self.corpus[doc["_id"]] = {
                "text": doc.get("text"),
                "title": doc.get("title")
            }

    def _load_queries(self):
        self.query_file.seek(0)  # Reset the StringIO object to the beginning
        for line in self.query_file:
            query = json.loads(line)
            self.queries[query["_id"]] = query.get("text")

    def _load_qrels(self):
        self.qrels_file.seek(0)  # Reset the StringIO object to the beginning
        reader = csv.reader(self.qrels_file, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            query_id, corpus_id, score = row[0], row[1], int(row[2])
            if query_id not in self.qrels:
                self.qrels[query_id] = {corpus_id: score}
            else:
                self.qrels[query_id][corpus_id] = score

def load_data_from_hf(task_name):
    try:
        queries_corpus_dataset = load_dataset(f"CoIR-Retrieval/{task_name}-queries-corpus")
        qrels_dataset = load_dataset(f"CoIR-Retrieval/{task_name}-qrels")

        corpus_data = queries_corpus_dataset['corpus']
        query_data = queries_corpus_dataset['queries']
        qrels_data = qrels_dataset['test']

        data_loader = InMemoryDataLoader(corpus_data, query_data, qrels_data)
        return data_loader.load_custom()
    except Exception as e:
        logger.error(f"Failed to load data for task {task_name}: {e}")
        return None

def get_tasks(tasks: list):
    all_tasks = {}

    # Define sub-tasks for special cases
    special_tasks = {
        "codesearchnet": [
            "CodeSearchNet-go", "CodeSearchNet-java", "CodeSearchNet-javascript",
            "CodeSearchNet-ruby", "CodeSearchNet-python", "CodeSearchNet-php"
        ],
        "codesearchnet-ccr": [
            "CodeSearchNet-ccr-go", "CodeSearchNet-ccr-java", "CodeSearchNet-ccr-javascript",
            "CodeSearchNet-ccr-ruby", "CodeSearchNet-ccr-python", "CodeSearchNet-ccr-php"
        ]
    }

    for task in tasks:
        if task in special_tasks:
            for sub_task in special_tasks[task]:
                task_data = load_data_from_hf(sub_task)
                if task_data is not None:
                    all_tasks[sub_task] = task_data
        else:
            task_data = load_data_from_hf(task)
            if task_data is not None:
                all_tasks[task] = task_data

    return all_tasks