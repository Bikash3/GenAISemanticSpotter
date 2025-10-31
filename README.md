# Automatic Ticket Classification
> Topic modelling and supervised classification of customer complaint tickets to assign each ticket to its relevant product/service department.

## Table of Contents
* [General Info](#general-information)
* [Stepwise Process](#stepwise-process)
  * [Step 1: Objective](#step-1---objective)
  * [Step 2: Data Loading](#step-2---data-loading)
  * [Step 3: Text Preparation](#step-3---text-preparation)
  * [Step 4: Exploratory Data Analysis (EDA)](#step-4---exploratory-data-analysis-eda)
  * [Step 5: Feature Extraction](#step-5---feature-extraction)
  * [Step 6: Topic Modelling](#step-6---topic-modelling)
  * [Step 7: Supervised Model Building](#step-7---supervised-model-building)
  * [Step 8: Model Training, Evaluation and Inference](#step-8---model-training-evaluation-and-inference)
* [Technologies Used](#technologies-used)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## General Info
> This repository contains a pipeline to process customer complaint records in JSON format, discover latent topic groups using matrix factorization, map topics to operational departments, and build supervised classifiers to assign new tickets to departments.

Input data: JSON file of complaint records (contains raw complaint text and related metadata).  
Final outputs: topic-labelled dataset and trained classifiers for automated ticket routing.

## Stepwise Process

### Step 1 - Objective
- Extract meaningful categories from unlabeled complaint text using unsupervised topic modelling.
- Map discovered topics to five operational classes:
  1. Credit card / Prepaid card
  2. Bank account services
  3. Theft / Dispute reporting
  4. Mortgages / Loans
  5. Others
- Build supervised classifiers using the derived labels to predict the category for new complaints.

### Step 2 - Data Loading
1. Read the JSON file and normalize into a tabular format (pandas DataFrame).
2. Inspect columns and basic schema to identify fields of interest (complaint text, product, dates, metadata).
3. Remove rows with empty complaint text to retain only analyzable records.

### Step 3 - Text Preparation
1. Clean raw text:
   - Lowercase conversion.
   - Remove text in square brackets.
   - Remove punctuation.
   - Remove tokens containing digits.
2. Lemmatize the cleaned text using a language model (spaCy small English model).
3. POS filtering:
   - Keep only nouns (POS tag == "NN") from lemmatized text to focus on topic-bearing tokens.
4. Post-process:
   - Remove masked personal tokens (e.g., 'xxxx').
   - Produce final cleaned corpus column for modelling.

### Step 4 - Exploratory Data Analysis (EDA)
1. Visualize complaint lengths (character distribution) to understand typical document size.
2. Create a word cloud of the top 40 tokens from the cleaned corpus to inspect frequent terms.
3. Compute and display top unigrams, bigrams and trigrams by frequency (top 30) to reveal common phrases.

### Step 5 - Feature Extraction
1. Initialize a vectorizer (TF-IDF) with sensible thresholds:
   - max_df (e.g., 0.95) to ignore very frequent terms.
   - min_df (e.g., 2) to ignore very rare tokens.
2. Fit the TF-IDF vectorizer on the cleaned corpus and compute the document-term matrix (DTM).
3. Use the DTM as input for topic modelling.

### Step 6 - Topic Modelling
1. Use Non-Negative Matrix Factorization (NMF) on the TF-IDF DTM.
2. Determine number of topics by manual inspection and trial (example uses n_components = 5).
3. For each topic, list top terms (example: top 15 words per topic) and inspect coherence.
4. Assign each document the dominant topic (argmax over NMF transform output).
5. Map numeric topic ids to human-readable topic names:
   - 0: Bank Account services
   - 1: Credit card or prepaid card
   - 2: Others
   - 3: Theft/Dispute Reporting
   - 4: Mortgage/Loan
6. Validate topic assignments by sampling complaints per topic and adjusting mapping if required.

### Step 7 - Supervised Model Building
1. Prepare labeled training data:
   - Keep columns: original complaint text and assigned topic label.
2. Feature pipeline for supervised models:
   - Create Count Vectorizer (or reuse TF-IDF as required) with appropriate min_df / max_df.
   - Transform complaint text to numeric feature vectors.
3. Models to try (minimum three):
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - (Optional) Multinomial Naive Bayes
4. Split data into train and test sets (example: 80/20 random split with a fixed random state).
5. Train each model on training set.

### Step 8 - Model Training, Evaluation and Inference
1. Evaluate models on test set using standard classification metrics:
   - Accuracy
   - Precision (weighted)
   - Recall (weighted)
   - F1-score (weighted)
   - Confusion matrix and per-class reports
2. Compare model performance and select the best model for deployment or further tuning.
3. For inference:
   - Apply the same text cleaning, lemmatization and POS-filtering steps.
   - Vectorize with the trained vectorizer.
   - Predict topic label using the trained classifier and map to topic name.

## Technologies Used
- Python
- pandas, NumPy
- spaCy (en_core_web_sm) for lemmatization and POS tagging
- scikit-learn (CountVectorizer, TfidfVectorizer, NMF, train_test_split, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, MultinomialNB)
- matplotlib, seaborn, wordcloud, plotly for visualization
- Jupyter Notebook

## Acknowledgements
- Assignment prepared as part of course exercises.
- Reference material:
  - scikit-learn documentation
  - spaCy documentation
  - wordcloud and visualization guides

## Contact
### Created by
  * Bikash Sarkar