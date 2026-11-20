# Natural Language Processing — Course Projects

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=black) ![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

My solutions to the assignments of the **Natural Language Processing** course at Amirkabir University of Technology (Tehran Polytechnic), M.Sc. in Artificial Intelligence and Robotics (Spring 2025) · Instructor: **Dr. Saeedeh Momtazi**.

Each folder contains the code (Jupyter notebooks / Python scripts) and the full written report with results and analysis.

## Assignments

| Assignment | Topics | Main tools | Files |
|---|---|---|---|
| [Language Models & Word Embeddings](HW1-Language-Models-and-Word-Embeddings) | N-gram language models, Word2Vec embeddings, embedding-based text classification | gensim, PyTorch | [code](HW1-Language-Models-and-Word-Embeddings/code) · [report](HW1-Language-Models-and-Word-Embeddings/report.pdf) |
| [Named Entity Recognition & POS Tagging](HW2-Named-Entity-Recognition-and-POS-Tagging) | NER evaluation with spaCy on resumes; POS tagging with a GRU vs. rule-based and most-frequent-tag baselines | spaCy, PyTorch, Hugging Face Datasets | [code](HW2-Named-Entity-Recognition-and-POS-Tagging/code) · [report](HW2-Named-Entity-Recognition-and-POS-Tagging/report.pdf) |
| [Large Language Models & Prompt Tuning](HW3-Large-Language-Models-and-Prompt-Tuning) | Prompting LLMs, fine-tuning, soft prompt tuning, efficient decoding | Transformers, PyTorch | [code](HW3-Large-Language-Models-and-Prompt-Tuning/code) · [report](HW3-Large-Language-Models-and-Prompt-Tuning/report.pdf) |

## How to run

```bash
git clone https://github.com/Aidin-Khalili/Natural-Language-Processing-Course-Projects.git
cd Natural-Language-Processing-Course-Projects
pip install numpy pandas scikit-learn torch transformers datasets gensim spacy seaborn matplotlib
jupyter notebook
```

Most notebooks were run on Google Colab or locally; paths to data files may need to be changed to your own.

## Notes

- Reports are written in Persian.
- Large datasets, model weights and course-provided input files are not included. Some figures in the notebooks were compressed or removed to keep the repository small; the full set is in each `report.pdf`.

## License

Released under the [MIT License](LICENSE): you are free to use, modify and share this code with attribution.

## Author

**Aidin Khalili** — [GitHub](https://github.com/Aidin-Khalili)
