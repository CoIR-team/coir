<h1 align="center">
<img style="vertical-align:middle" width="400" src="pictures/coir_3.png" />
</h1>

<p align="center">
    <a href="https://github.com/beir-cellar/beir/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/beir-cellar/beir.svg">
    </a>
    <a href="https://www.python.org/">
            <img alt="Build" src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple">
    </a>
    <a href="https://github.com/CoIR-team/coir/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/beir-cellar/beir.svg?color=green">
    </a>
    <a href="xxxxxxxxxx">
        <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
    <a href="https://github.com/CoIR-team/coir">
        <img alt="Downloads" src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103">
    </a>
</p>



# What is CoIR?

**CoIR** (**Co**de **I**nformation **R**etrieval) benchmark, is designed to evaluate code retrieval capabilities. CoIR includes **10** curated code datasets, covering **8** retrieval tasks across **7** domains. It also provides a **common and easy** Python framework, installable via pip, and shares the same data schema as benchmarks like MTEB and BEIR for easy cross-benchmark evaluations.

For **models and datasets**, checkout out **Hugging Face (HF)** page: [https://huggingface.co/CoIR-Retrieval](https://huggingface.co/CoIR-Retrieval).

For more information, checkout out our **publication**: [COIR: A Comprehensive Benchmark for Code Information Retrieval Models](linktobedo)

<div align="center">
    <img src="pictures/coir_overview.svg" width="850" />
    <br>
    <strong>Overview of COIR benchmark.</strong>
</div>



# Data Availability

All data has been uploaded to our Hugging Face page: [CoIR-Retrieval](https://huggingface.co/CoIR-Retrieval)


### Statistics of datasets in coir benchmark
\# is the quantity of query/corpus instances. L refers to the average numbers of words per query/corpus. Datasets marked by \(^\dag\) are created by us.

| **Main Task**                | **Sub Task**                       | **Domain**     | **Dataset**                  | **Language**                                   | **#Query (train/dev/test)** | **#Corpus** | **L_Query** | **L_Corpus** |
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



## Simple Use（python>3.8）

1. Download this GitHub repository

    ```python
    import coir
    from coir.models import YourCustomDEModel
    
    model_name = "intfloat/e5-base-v2"
    
    # Load the model
    model = YourCustomDEModel(model_name=model_name)
    
    # Get tasks
    #all task ["codetrans-dl","stackoverflow-qa","apps","codefeedback-mt","codefeedback-st","codetrans-contest","synthetic-text2sql","cosq"]
    tasks = coir.get_tasks(tasks=["codetrans-dl"])
    
    # Initialize evaluation
    evaluation = coir.COIR(tasks=tasks)
    
    # Run evaluation
    results = evaluation.run(model, output_folder=f"results/{model_name}")
    print(results)
    ```



## Using the Python Package

1. Install the `coir-eval` package

    ```bash
    pip install coir-eval
    ```

2. Use the following code to run the evaluation

    ```python
    import coir
    from coir.data_loader import get_tasks
    from coir.evaluation import COIR
    from coir.models import YourCustomDEModel
    
    model_name = "intfloat/e5-base-v2"
    
    # Load the model
    model = YourCustomDEModel(model_name=model_name)
    
    # Get tasks
    #all task ["codetrans-dl","stackoverflow-qa","apps","codefeedback-mt","codefeedback-st","codetrans-contest","synthetic-text2sql","cosq"]
    tasks = get_tasks(tasks=["codetrans-dl"])
    
    # Initialize evaluation
    evaluation = COIR(tasks=tasks)
    
    # Run evaluation
    results = evaluation.run(model, output_folder=f"results/{model_name}")
    print(results)
    ```



## Disclaimer

Similar to Tensorflow [datasets](https://github.com/tensorflow/datasets) or Hugging Face's [datasets](https://github.com/huggingface/datasets) library, we just downloaded and prepared public datasets. We only distribute these datasets in a specific format, but we do not vouch for their quality or fairness, or claim that you have license to use the dataset. It remains the user's responsibility to determine whether you as a user have permission to use the dataset under the dataset's license and to cite the right owner of the dataset.

If you're a dataset owner and wish to update any part of it, or do not want your dataset to be included in this library, feel free to post an issue here or make a pull request!

If you're a dataset owner and wish to include your dataset or model in this library, feel free to post an issue here or make a pull request!


## Citing & Authors
If you find this repository helpful, feel free to cite our publication [COIR: A Comprehensive Benchmark for Code Information Retrieval Models](xxx):

```
@misc{
    xxx
}
```
