# Semantic Spotter
> Document question-answer pipeline that indexes PDFs, retrieves relevant passages and returns source-referenced answers using a vector index and an external text model.

## Table of Contents
* [General Info](#general-information)
* [Stepwise Process](#stepwise-process)
  * [Step 1: Import Libraries](#step-1---import-the-necessary-libraries)
  * [Step 2: Drive Mounting and API Key](#step-2---mount-drive-and-set-api-key)
  * [Step 3: Data Loading (Ingestion)](#step-3---data-loading-ingestion)
  * [Step 4: Build Query Engine (Splitter, Embedding, Index)](#step-4---build-query-engine-splitter-embedding-index)
  * [Step 5: Response Pipeline](#step-5---response-pipeline)
  * [Step 6: Interactive Conversation Flow](#step-6---interactive-conversation-flow)
  * [Step 7: Testing Pipeline](#step-7---testing-pipeline)
  * [Step 8: Evaluation of Responses](#step-8---evaluation-of-responses)
  * [Step 9: Presentation and Usage](#step-9---presentation-and-usage)
  * [Step 10: Deployment Notes](#step-10---deployment-notes)
* [Technologies Used](#technologies-used)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## General Info
> Semantic Spotter ingests PDF documents, splits text into searchable segments, stores vector representations in a similarity index, and exposes a simple query pipeline that returns an answer plus provenance (file name and page labels). The implementation includes utilities for automated testing and three evaluators to measure relevance, correctness and faithfulness of answers.

Primary outputs:
- Source-referenced answers for document queries.
- Test-suite output (tabular) for evaluation questions.
- Evaluator feedback (relevance, correctness, faithfulness).

## Stepwise Process

### Step 1 - Import the necessary libraries
1. Install required packages for indexing, embedding, and model access.
2. Import Python modules:
   - os, random, pandas, openai, pathlib, IPython.display
   - Indexing and reader components (SentenceSplitter, PDFReader, VectorStoreIndex, Document)
   - Embedding and model connectors
   - Evaluator classes and dataset generator for automated tests

### Step 2 - Mount drive and set API key
1. Mount the drive that holds the documents (example uses Google Drive in notebook).
2. Load the external service API key (securely) and set as environment variable and client key.
3. Verify model connectivity by issuing a small test chat request.

Step-by-step:
- drive.mount('/content/drive')
- set openai.api_key from secure source (environment or secret store)
- test a sample chat to confirm the client works

### Step 3 - Data Loading (Ingestion)
1. Use a PDF reader to load PDF files as document objects that preserve metadata (file_name, page_label).
2. Example:
   - loader = PDFReader()
   - document1 = loader.load_data(file = '/path/to/Policy.pdf')
3. Confirm loaded document count and inspect a document object to verify text and metadata presence.

Notes:
- Preserve page boundaries and metadata so provenance can be returned with answers.

### Step 4 - Build query engine (Splitter, Embedding, Index)
1. Configure segmentation:
   - SentenceSplitter or similar with chunk_size and chunk_overlap (example: 512 / 20).
2. Configure embedding model:
   - Instantiate OpenAIEmbedding (or other connector) with chosen model name.
3. Configure text model:
   - Create llm instance with deterministic settings (temperature=0, max_tokens limited).
4. Index construction:
   - Build a VectorStoreIndex.from_documents(document_list).
   - Create query engine and chat engine:
     - query_engine = index.as_query_engine(similarity_top_k=3)
     - chat_engine = index.as_chat_engine(chat_mode="condense_plus_context", verbose=True)

Tuning tips:
- Adjust chunk_size / overlap for recall vs context size.
- Tune similarity_top_k to balance supporting evidence vs context budget.

### Step 5 - Response Pipeline
1. Implement a helper that accepts a user query and returns:
   - response text
   - top provenance: file name and page labels
2. Example helper pattern:
   - response = query_engine.query(user_input)
   - file_name = response.source_nodes[0].node.metadata['file_name']
   - page_numbers = [node.node.metadata['page_label'] ...]
   - return { "response": response.response, "file_name": file_name, "page_number": page_numbers }

3. Encapsulate this in query_response(user_input) for reuse by interactive and testing flows.

### Step 6 - Interactive Conversation Flow
1. Provide a console interaction helper that:
   - Reads user input in a loop
   - Calls chat_engine.chat(user_input) or query_response(user_input)
   - Displays both the user question and assistant answer
   - Exits when user types 'exit'
2. Use deterministic model parameters (temperature=0) for reproducible outputs during manual inspection.

### Step 7 - Testing Pipeline
1. Create a testing function that accepts a list of questions, queries the index and returns a pandas DataFrame with columns:
   - Question, Response, Page
2. Example:
   - For each question: res = query_response(q); collect (q, res['response'], ', '.join(res['page_number']))
   - feedback_df = pd.DataFrame(...)

Purpose:
- Quickly generate batch outputs for manual review and evaluator input.

### Step 8 - Evaluation of Responses
1. Generate evaluation questions automatically:
   - Use DatasetGenerator.from_documents(document_list).generate_questions_from_nodes()
2. Use three evaluators:
   - RelevancyEvaluator: measures whether the answer addresses the query.
   - CorrectnessEvaluator: compares answer against a reference passage.
   - FaithfulnessEvaluator: checks whether answer is supported by source nodes (detects unsupported assertions).
3. Sample workflow:
   - Pick question, obtain response via query_engine.query
   - Call each evaluator and collect score, passing boolean and feedback

Notes:
- Use evaluator feedback to identify frequent failure modes (missing provenance, hallucinated claims, partial answers).

### Step 9 - Presentation and Usage
1. Console usage:
   - initialize_conv() to run an interactive loop for queries.
2. Notebook usage:
   - Use IPython.display to show user questions and assistant answers cleanly.
3. Batch testing:
   - testing_pipeline(questions) to produce a DataFrame for reporting.

Return format:
- Each answer should be delivered with file_name and page_number fields to enable traceability.

### Step 10 - Deployment Notes
1. Secrets:
   - Never hardcode API keys; use environment variables or a secret manager.
2. Scalability:
   - Persist the vector index to disk or use a vector database for larger corpora.
   - Cache embeddings and index artifacts to reduce cost and latency.
3. Robustness:
   - Add OCR step for scanned PDFs (Tesseract or commercial OCR) before indexing.
   - Implement retry/backoff and rate-limit handling for external service calls.
4. Monitoring:
   - Log query latencies, model errors, and evaluator scores for continuous improvement.
5. Reproducibility:
   - Keep deterministic model settings for evaluation runs (temperature=0, fixed random seeds).

## Technologies Used
- Python 3.x
- OpenAI client (or configured text model client)
- Llama-Index components (SimpleDirectoryReader, PDFReader, VectorStoreIndex, SentenceSplitter)
- pandas
- IPython.display utilities
- Evaluation utilities (RelevancyEvaluator, CorrectnessEvaluator, FaithfulnessEvaluator)
- Google Drive (for notebook examples that mount drive)

## Acknowledgements
- Implementation demonstrates a document retrieval + response pipeline with provenance and automated evaluators for quality checks.
- Notebook code adapted to illustrate indexing, query, automated tests and three evaluators.

## Contact
### Created by
  * Bikash Sarkar
  * (Use repository metadata for additional contributors)
