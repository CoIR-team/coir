# coir



### Statistics of datasets in \name benchmark
\# is the quantity of query/corpus instances. \(L_{(\cdot)}\) refers to the average numbers of words per query/corpus. Datasets marked by \(^\dag\) are created by us.

| **Main Task**                | **Sub Task**                       | **Domain**     | **Dataset**                  | **Language**                                   | **#Query (train/dev/test)** | **#Corpus** | **\(L_{\text{Query}}\)** | **\(L_{\text{Corpus}}\)** |
|------------------------------|------------------------------------|----------------|------------------------------|------------------------------------------------|-----------------------------|-------------|-------------------------|---------------------------|
| Text-to-Code Retrieval       | Code Contest Retrieval             | Code Contest   | APPS                         | py                                             | 5k/-/3.8K                   | 9K          | 1.4K                    | 575                       |
|                              | Web Query to Code Retrieval        | Web query      | CosQA                        | py                                             | 19k/-/500                   | 21K         | 37                      | 276                       |
|                              | Text to SQL Retrieval              | Database       | Synthetic Text2SQL           | sql                                            | 100k/-/6K                   | 106K        | 83                      | 127                       |
| Code-to-Text Retrieval       | Code Summary Retrieval             | Github         | CodeSearchNet                | go, java, js, php, py, ruby                    | 905k/41k/53K                | 1M          | 594                     | 156                       |
| Code-to-Code Retrieval       | Code Context Retrieval             | Github         | CodeSearchNet-CCR^\dag       | go, java, js, php, py, ruby                    | 905k/41k/53K                | 1M          | 154                     | 113                       |
|                              | Similar Code Retrieval             | Deep Learning  | CodeTrans Ocean-DL           | py                                             | 564/72/180                  | 816         | 1.6K                    | 1.5K                      |
|                              |                                    | Contest        | CodeTrans Ocean-Contest      | c++, py                                        | 561/226/446                 | 1K          | 770                     | 1.5K                      |
| Hybrid Code Retrieval        | Single-turn Code QA                | Stack Overflow | StackOverflow QA^\dag        | miscellaneous                                  | 13k/3k/2K                   | 20K         | 1.4K                    | 1.2K                      |
|                              |                                    | Code Instruction | CodeFeedBack-ST              | html, c, css, sql, js, sql, py, shell, ruby, rust, swift | 125k/-/31K | 156K        | 722                     | 1.5K                      |
|                              | Multi-turn Code QA                 | Code Instruction | CodeFeeback-MT               | miscellaneous                                  | 53k/-/13K                   | 66K         | 4.4K                    | 1.5K                      |




# simple use
```python
import coir
from coir.models import YourCustomDEModel

model_name = "intfloat/e5-base-v2"

# Load the model
model = YourCustomDEModel(model_name=model_name)

# Get tasks
tasks = coir.get_tasks(tasks=["stackoverflow-qa"])

# Initialize evaluation
evaluation = coir.COIR(tasks=tasks)

# Run evaluation
results = evaluation.run(model, output_folder=f"results/{model_name}")
print(results)
```

