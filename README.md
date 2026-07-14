
Customer Review Intelligence

An end-to-end natural language processing project that analyzes Amazon Fine Food Reviews to classify customer sentiment, interpret model predictions, and identify common complaint themes using topic modeling.

## Project Overview

Customer reviews contain more than star ratings. They reveal what customers like, what frustrates them, and where a business can improve. This project uses NLP and machine learning to turn review text into actionable insight.

The project focuses on three main questions:

- Can review text predict whether feedback is negative, neutral, or positive?
- Which model performs best for sentiment classification?
- What themes appear most often in negative customer reviews?

## Dataset

The project uses the [Amazon Fine Food Reviews dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) from Kaggle.

The dataset includes:

- Review text
- Review summary
- Star rating
- Product and user IDs
- Helpfulness votes
- Review timestamp

Star ratings were converted into sentiment labels:

- `1-2 stars`: negative
- `3 stars`: neutral
- `4-5 stars`: positive

## Tools and Libraries

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- scikit-learn
- KaggleHub

## Project Workflow

1. Loaded and inspected the Amazon Fine Food Reviews dataset
2. Cleaned review text using lowercasing, punctuation removal, stopword removal, and lemmatization
3. Created sentiment labels from star ratings
4. Performed exploratory data analysis on sentiment, ratings, review length, helpfulness, and common words
5. Trained a baseline TF-IDF + Logistic Regression model
6. Compared Logistic Regression, Multinomial Naive Bayes, and Linear SVC
7. Interpreted model predictions using TF-IDF feature weights
8. Built a binary sentiment model to test positive-vs-negative classification
9. Used LDA topic modeling to identify themes in negative reviews
10. Summarized business insights and recommendations

## Model Results

The best three-class sentiment model was **Linear SVC**.

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| Linear SVC | 0.849 | 0.849 | 0.658 |
| Logistic Regression | 0.804 | 0.824 | 0.638 |
| Multinomial Naive Bayes | 0.809 | 0.744 | 0.403 |

Linear SVC performed strongly on positive reviews and improved overall performance compared with the baseline model. Neutral reviews remained the hardest class to predict because they often contain mixed or mild language.

## Binary Sentiment Model

Because neutral reviews were difficult to classify, a second model was trained using only positive and negative reviews.

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| Linear SVC Binary Classifier | 0.918 | 0.920 |

This showed that the model can distinguish positive and negative reviews much more reliably than it can classify neutral reviews.

## Topic Modeling Insights

LDA topic modeling was applied to negative reviews to identify common complaint themes.

| Complaint Theme | Share of Negative Review Sample |
|---|---:|
| Amazon ordering, packaging, and delivery issues | 29.3% |
| Flavor and taste dissatisfaction | 27.8% |
| Ingredients, sugar, and snack taste complaints | 16.3% |
| Pet food and pet treat complaints | 13.7% |
| Coffee, price, and value concerns | 12.2% |
| Tea and dietary product complaints | 0.7% |

The most common negative-review themes were fulfillment or packaging issues and taste-related dissatisfaction. This suggests that customer dissatisfaction is not only related to the product itself, but also to delivery experience, product condition, and customer expectations.

## Key Takeaways

- Positive reviews dominate the dataset, creating class imbalance.
- Linear SVC was the best-performing model for three-class sentiment classification.
- Neutral sentiment was the most difficult class because neutral reviews often contain mixed feedback.
- A binary positive-vs-negative model performed much better than the three-class model.
- Topic modeling helped translate negative sentiment into concrete business themes.
- The largest complaint areas were ordering, packaging, delivery, and taste dissatisfaction.

## Business Recommendations

- Monitor negative-review topics regularly to detect recurring customer pain points.
- Improve packaging and delivery consistency to reduce fulfillment-related complaints.
- Investigate taste and flavor complaints for products with repeated negative feedback.
- Treat neutral predictions with caution because they are less reliable than positive or negative predictions.
- Use positive-review language to identify product strengths and marketing themes.

## Limitations and Future Work

- The modeling workflow used a sample of the full dataset for faster experimentation.
- Sentiment labels were created from star ratings, which may not perfectly capture the emotional tone of each review.
- Neutral reviews were underrepresented and linguistically ambiguous.
- Future work could include hyperparameter tuning, transformer-based models such as BERT, trend analysis over time, and deployment as a Streamlit dashboard.

## Repository Structure

```text
customer-review-intelligence/
  notebooks/
    customer-review-intelligence.ipynb
  README.md
  requirements.txt
  .gitignore
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook notebooks/customer-review-intelligence.ipynb
```

The dataset can be downloaded from Kaggle using KaggleHub or attached directly in a Kaggle notebook environment.
