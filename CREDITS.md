# Credits & Citations

This project adheres to academic and professional integrity standards. All external tools, datasets, algorithms, and libraries are explicitly attributed below:

---

## 1. Dataset
- **Customer Support on Twitter (`twcs.csv`)**:
  - **Author**: ThoughtVector ([Kaggle Dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter))
  - **Description**: Public dataset containing ~2.81 million customer support tweets across diverse brands on Twitter, including multi-turn conversational threads.
  - **Brand Surface**: Specifically filtered to `@AskPlayStation` (PlayStation customer support), containing 19,098 brand tweets and 18,407 reconstructed conversation threads.
  - **License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

---

## 2. LLM & Provider SDK
- **Google Gemini API**:
  - **SDK**: `google-genai` / `google-generativeai` by Google DeepMind.
  - **Models**: `gemini-2.5-flash` for zero-shot structured intent classification, grounded response drafting, and QA judge rubric evaluation.

---

## 3. Libraries & Algorithms
- **Information Retrieval Engine**:
  - Implementation of inverted-index TF-IDF with sublinear term-frequency scaling and exact Cosine Similarity based on foundational IR principles from *Introduction to Information Retrieval* (Manning, Raghavan, Schütze, Cambridge University Press, 2008).
- **Scikit-Learn**:
  - Inter-annotator agreement metrics (`sklearn.metrics.cohen_kappa_score`) and confusion matrix visualization (`sklearn.metrics.confusion_matrix`).
- **Data Processing**:
  - `pandas` and `numpy` for thread indexing and metric aggregation.

---

## 4. Prompts & Evaluation Framework
- The 4-dimensional QA Rubric (Groundedness, Technical Correctness, Brand Tone, Actionability) was adapted from standard conversational AI evaluation methodologies (e.g., MT-Bench and G-Eval frameworks) and customized specifically to Sony Interactive Entertainment / PlayStation Customer Support service level agreements.
