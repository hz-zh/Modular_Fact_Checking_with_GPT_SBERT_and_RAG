# Modular Fact Checking with GPT, sBERT, and RAG

This repository contains the code and resources for a modular fact-checking system designed to verify claims using evidence retrieved from external knowledge sources (primarily Wikipedia). The system leverages Large Language Models (specifically OpenAI's GPT-4o-mini), Sentence-BERT (sBERT), and principles of Retrieval-Augmented Generation (RAG) to create a multi-stage verification pipeline.

## Overview

The primary goal of this project is to develop and evaluate a system that can automatically:

1. Retrieve relevant documents based on a given claim.
2. Extract specific evidence sentences from those documents.
3. Classify the claim's stance (Supports, Refutes, Not Enough Info) based on the extracted evidence.

The system is designed modularly, allowing for the evaluation and fine-tuning of individual components.

## Key Features & Modules

* **Modular Architecture:** The system is broken down into distinct modules for document retrieval, evidence extraction/distillation, and final classification (See `system_dev_v4.ipynb`).
* **Hybrid Evidence Extraction:** Combines sBERT similarity-based pre-filtering with GPT-4o-mini's contextual understanding for precise sentence selection.
* **Fine-tuning:** Includes notebooks for fine-tuning GPT-4o-mini and sBERT models on specific sub-tasks (query generation, sentence extraction, classification) using the FEVER dataset.
* **Claim Rephrasing:** Explores techniques for iteratively rephrasing claims to potentially improve evidence retrieval (`Fine-tuning_claim-rephrase_tests.ipynb`).
* **Claim Analysis:** Investigates linguistic features of claims (e.g., syntactic complexity) to understand potential correlations with verification difficulty (`claim_analysis.ipynb`).
* **Evaluation:** Uses the official FEVER scorer and standard metrics (Strict Score, Label Accuracy, Evidence F1) for performance assessment. Includes evaluation against the LIAR dataset (`liar_gpt4omini_base_eval.ipynb`).

## Technologies Used

* Python 3.x
* Jupyter Notebooks
* OpenAI API (GPT-4o-mini)
* `sentence-transformers` (sBERT)
* `wikipedia` library
* `nltk`
* `spacy`
* `pandas`
* `scikit-learn`
* FEVER Scorer (`fever-scorer`)

## Datasets

* **FEVER (Fact Extraction and VERification):** The primary dataset used for fine-tuning and evaluation (`datasets/FEVER`). (Thorne et al., 2018)
* **LIAR:** Used for benchmarking/evaluating base model performance (`liar_gpt4omini_base_eval.ipynb`). (Wang, 2017)

## Repository Structure & Notebook Descriptions

* `datasets/FEVER/`: Contains FEVER dataset related files, including raw data, processed tabular sets, and JSONL files for GPT fine-tuning.
* `.gitignore`: Specifies intentionally untracked files that Git should ignore.
* `FEVER_set_creation.ipynb`: Notebook for initial processing of the FEVER dataset, associating claims with evidence, generating tabular formats, and creating JSONL sets.
* `FEVER_tabular_set_update.ipynb`: Updates or further processes the tabular FEVER datasets (e.g., adding complexity metrics).
* `Fine-tuning_claim-rephrase_tests.ipynb`: Contains code related to testing and potentially fine-tuning models for claim rephrasing strategies within Module 2.
* `GPT_clf_fine_tuning.ipynb`: Fine-tunes a GPT-4o-mini model for the stance classification task (Module 3).
* `GPT_query_fine_tuning.ipynb`: Fine-tunes a GPT-4o-mini model for generating Wikipedia page title queries (Module 1).
* `GPT_sentEx_fine_tuning.ipynb`: Fine-tunes a GPT-4o-mini model for extracting evidence sentences (Module 2).
* `claim_analysis.ipynb`: Performs analysis on claim characteristics (e.g., syntactic/semantic complexity).
* `liar_gpt4omini_base_eval.ipynb`: Evaluates the base GPT-4o-mini model, likely on classification, using the LIAR dataset.
* `sBERT_sentEx_fine_tuning.ipynb`: Fine-tunes an sBERT model for sentence extraction/filtering (Module 2).
* `system_dev_v4.ipynb`: The main script for developing, integrating, and evaluating the complete modular fact-checking system (latest version).
* `test_set_creation.ipynb`: Creates specific test subsets or formats data for evaluation purposes.

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/hz-zh/Modular_Fact_Checking_with_GPT_SBERT_and_RAG.git
    cd Modular_Fact_Checking_with_GPT_SBERT_and_RAG
    ```

2. Install dependencies (ensure you have Python 3.8+):

    ```bash
    pip install -r requirements.txt # Assuming a requirements.txt exists or listing key packages
    pip install sentence-transformers wikipedia nltk spacy pandas openai scikit-learn rouge-score
    # Install specific versions if needed
    ```

3. Install the FEVER Scorer (follow instructions within `system_dev_v4.ipynb` or the official FEVER scorer repository if needed).
4. Download necessary NLTK and spaCy data (commands often included within notebooks):

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    # ... other required NLTK packages
    import spacy
    spacy.download('en_core_web_sm')
    ```

5. Set up your OpenAI API key (e.g., as an environment variable or using Google Colab secrets).

## Usage

* Explore individual notebooks for data preparation, fine-tuning, and analysis steps.
* Run `system_dev_v4.ipynb` to execute the full fact-checking pipeline and evaluation on test data. Adjust parameters within the notebook for different configurations.

## Evaluation

System performance is evaluated using the official FEVER scorer (`fever-scorer`), calculating metrics such as:

* Strict Score (requires both correct label and correct evidence)
* Label Accuracy
* Evidence Precision, Recall, and F1-score

Detailed results and analysis for different system configurations can be found within the report generated by `system_dev_v4.ipynb` and discussed in the accompanying project report/documentation.

## Author

* Henry Zelenak | [hz-zh](https://github.com/hz-zh)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
