import os
import json
import logging
from coir.beir.retrieval.evaluation import EvaluateRetrieval
from coir.beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

logger = logging.getLogger(__name__)

class COIR:
    def __init__(self, tasks):
        self.tasks = tasks

    def run(self, model, output_folder: str):
        results = {}
        for task_name, task_data in self.tasks.items():
            corpus, queries, qrels = task_data

            # Initialize custom model
            custom_model = DRES(model, batch_size=128)
            retriever = EvaluateRetrieval(custom_model, score_function="cos_sim")

            # Retrieve results
            task_results = retriever.retrieve(corpus, queries)

            # Evaluate results
            ndcg, _map, recall, precision = retriever.evaluate(qrels, task_results, retriever.k_values)
            metrics = {
                "NDCG": ndcg,
                "MAP": _map,
                "Recall": recall,
                "Precision": precision
            }

            # Save results
            os.makedirs(output_folder, exist_ok=True)
            with open(os.path.join(output_folder, f"{task_name}.json"), 'w') as json_file:
                json.dump({"metrics": metrics}, json_file, indent=4)

            logger.info(f"Results for {task_name} saved to {output_folder}")
            results[task_name] = metrics
        return results
