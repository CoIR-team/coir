import unittest
from coir import COIR, get_tasks
from coir.models import YourCustomDEModel


class TestEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 在所有测试之前运行一次，加载模型和任务数据
        cls.model_name = "intfloat/e5-base-v2"
        cls.model = YourCustomDEModel(model_name=cls.model_name)
        cls.tasks = get_tasks(tasks=["stackoverflow-qa"])

    def test_initialization(self):
        evaluation = COIR(tasks=self.tasks)
        self.assertIsNotNone(evaluation, "Evaluation object should be initialized")

    def test_run_evaluation(self):
        evaluation = COIR(tasks=self.tasks)
        results = evaluation.run(self.model, output_folder=f"results/{self.model_name}")
        self.assertIsNotNone(results, "Results should not be None")
        self.assertTrue(len(results) > 0, "Results should not be empty")


if __name__ == '__main__':
    unittest.main()