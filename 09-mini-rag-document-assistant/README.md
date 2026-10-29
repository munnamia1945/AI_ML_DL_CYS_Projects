# **Project 09: Mini RAG Document Assistant**

An educational Retrieval-Augmented Generation (RAG) system built with Python and Hugging Face models to retrieve relevant information from uploaded documents, generate context-based answers, and provide source references.

## **Project Overview**

This project implements a basic end-to-end RAG pipeline in Google Colab. Users can upload PDF or TXT documents, extract and split their text into chunks, generate embeddings, and retrieve relevant content through semantic similarity search. A text-generation model then uses the retrieved context to answer questions.

The system also includes a basic retrieval evaluation workflow and a template for manually reviewing answer quality.

## **Objectives**

* Build an end-to-end document question-answering pipeline.  
* Extract text from supported document formats.  
* Generate dense vector embeddings for document chunks.  
* Retrieve relevant text using semantic similarity.  
* Generate context-based answers with source references.  
* Explore retrieval evaluation and manual answer-quality assessment.

## **Key Features**

* PDF and TXT document-processing logic.  
* Overlapping, word-based text chunking.  
* Semantic search using sentence embeddings.  
* Context-based answer generation.  
* Source metadata, chunk identifiers, and similarity scores.  
* Basic Hit@4 retrieval evaluation.  
* CSV and JSON exports for inspection and evaluation.

## **Technologies Used**

| Technology | Purpose |
| ----- | ----- |
| Python | Main programming language |
| Google Colab | Development environment |
| PyTorch | Model execution |
| Sentence Transformers | Text embedding generation |
| Hugging Face Transformers | Answer generation |
| `pypdf` | PDF text extraction |
| Pandas and NumPy | Data processing |
| Matplotlib and Seaborn | Similarity-score visualization |

## **Models**

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`

* Generates 384-dimensional text embeddings.  
* Uses normalized embeddings for similarity-based retrieval.

**Answer-generation model:** `google/flan-t5-base`

* Generates answers using retrieved document context.  
* Uses deterministic generation settings in the provided implementation.

## **How It Works**

The pipeline follows these steps:

1. **Document ingestion:** Upload a TXT or PDF file.  
2. **Text extraction:** Read the text and retain available source metadata.  
3. **Chunking:** Divide extracted text into overlapping word-based chunks.  
4. **Embedding generation:** Convert chunks into numerical vectors.  
5. **Semantic retrieval:** Compare the query embedding with document embeddings and retrieve the highest-scoring chunks.  
6. **Answer generation:** Provide the retrieved context and question to the generation model.  
7. **Source reporting:** Display source filenames, chunk or page metadata, and similarity scores.  
8. **Evaluation and export:** Record retrieval evaluation results and export project artifacts.

### **Chunking Configuration**

| Parameter | Value |
| ----- | ----- |
| Chunk size | 180 words |
| Chunk overlap | 40 words |
| Default retrieval limit | Top 4 chunks |
| Embedding dimensions | 384 |

The chunking strategy uses a sliding window over whitespace-separated words.

## **Experimental Setup**

The recorded execution used the following test document:

* **File:** `python_rag_guide.txt`  
* **Reported file size:** 1,436 bytes  
* **Extracted text:** 1,411 characters  
* **Documents processed:** 1  
* **Chunks created:** 2

The test document contained introductory material about Python, machine learning, and model evaluation.

Although the implementation includes PDF text-extraction logic, the recorded experiment used a TXT file. PDF processing was therefore not empirically verified in this run.

## **How to Run**

1. Open the project notebook in Google Colab.  
2. Select a suitable runtime.  
3. Execute the notebook cells in order.  
4. Upload a text-based PDF or TXT file when prompted.  
5. Allow the notebook to extract, chunk, and embed the document.  
6. Enter a question that can be answered from the uploaded material.  
7. Review the generated answer, retrieved sources, and similarity scores.  
8. Inspect the exported files in the `project09_outputs` directory.

### **Install Dependencies**

The notebook installs the required packages using:

pip install pypdf sentence-transformers transformers accelerate pandas numpy matplotlib seaborn

PyTorch is used by the notebook for model execution. Available GPU support depends on the selected runtime.

## **Example Question and Answer**

**Question:**

What is the difference between precision and recall?

**Answer summary:**

* **Precision** measures how many predicted positive cases are actually positive.  
* **Recall** measures how many actual positive cases are correctly identified.

The recorded generated answer was truncated, so this example demonstrates the question-answering workflow rather than a fully validated answer-generation result.

**Retrieved source references:**

* `python_rag_guide.txt`, Chunk 2 — cosine similarity: 0.5095  
* `python_rag_guide.txt`, Chunk 1 — cosine similarity: 0.2763

These similarity scores indicate relative embedding similarity for the query and retrieved chunks. They are not probabilities of answer correctness.

## **Evaluation Methodology**

### **Retrieval Evaluation**

The notebook implements Hit@4. In this setup, the metric checks whether the expected source appears among the top four retrieved chunks.

**Recorded result:**

| Metric | Observed result |
| ----- | ----- |
| Hit@4 | 1.0 |
| Evaluation questions | 1 |
| Evaluation type | Preliminary retrieval check |

The result was obtained using a placeholder evaluation question and a single test document. Consequently, the score should not be interpreted as a reliable estimate of retrieval performance across diverse questions or documents.

### **Answer Quality Evaluation**

The notebook generates a CSV template for manually assessing:

* Answer relevance (1–5)  
* Answer groundedness (1–5)  
* Citation correctness (0 or 1\)

Manual answer-quality scoring was not completed in the recorded run. These metrics are **not evaluated**.

No validated answer-accuracy, precision, recall, or F1-score was established for the generated responses.

## **Generated Output Files**

The notebook exports the following artifacts to `project09_outputs`:

| File | Purpose |
| ----- | ----- |
| `document_chunks.csv` | Extracted chunks and associated metadata |
| `retrieval_evaluation.csv` | Retrieval evaluation records |
| `answer_quality_review.csv` | Template for manual answer-quality assessment |
| `example_rag_response.json` | Example question, generated answer, and source references |

## **Limitations**

* Only one short TXT document was evaluated.  
* The retrieval evaluation used a single placeholder question.  
* Manual answer-quality scores were not collected.  
* The example generated answer was truncated.  
* PDF text extraction was implemented but not tested in the recorded experiment.  
* Scanned PDFs require OCR, which is not included in the described pipeline.  
* Performance with large document collections has not been evaluated.  
* Retrieved context and source references do not guarantee that every generated claim is correct.

## **Future Improvements**

* Evaluate retrieval using a larger set of realistic questions with known supporting sources.  
* Test the pipeline on multiple TXT and PDF documents.  
* Complete manual answer-quality assessment using relevance, groundedness, and citation correctness.  
* Investigate answer truncation and improve generation settings.  
* Explore semantic or recursive chunking strategies.  
* Add more comprehensive retrieval and answer-quality metrics.  
* Evaluate performance on larger document collections.

## **Learning Outcomes**

Through this project, I gained practical experience with:

* Building a basic Retrieval-Augmented Generation pipeline.  
* Extracting and chunking document text.  
* Generating and comparing dense text embeddings.  
* Implementing semantic retrieval.  
* Passing retrieved context to a text-generation model.  
* Associating answers with source metadata.  
* Distinguishing preliminary retrieval evaluation from answer-quality evaluation.

## **Author**

**Md. Munna Mia**

CSE Undergraduate \| AI/ML and Cybersecurity Interests

GitHub: [munnamia1945](https://github.com/munnamia1945)

---

*This project is an educational prototype. The reported results reflect the limited experiment described above and should not be interpreted as evidence of production-level performance.*

