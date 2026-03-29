# Model Comparison Summary
**Customer Churn Prediction — Internship Assessment**

---

## What I Was Trying to Do

The goal was simple — figure out which customers are likely to leave, before they actually do.
I trained four different ML models on the Telco churn dataset and compared them honestly.
Not every model performed the way I expected, and I'll get into that below.

---

## The Dataset at a Glance

Before even touching a model, the first thing I noticed from the churn distribution plot was
that only **26.5% of customers actually churned** (1,869 out of 7,043). That means the
dataset is imbalanced — and if I didn't handle that, any model could just predict "No Churn"
for everyone and still hit 73% accuracy while being completely useless.

So I used **SMOTE** on the training set to balance the classes before fitting any model.

---

## The Four Models I Compared

I picked these four specifically because they represent a natural progression — from the
simplest linear approach all the way to a modern boosting algorithm:

| Model | Type |
|-------|------|
| Logistic Regression | Linear baseline |
| Decision Tree | Rule-based splits |
| Random Forest | Ensemble of trees |
| XGBoost | Sequential boosting |

---

## Actual Results (From My Runs)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7393 | 0.5044 | 0.7655 | 0.6081 | 0.832 |
| Decision Tree | 0.7037 | 0.4632 | 0.7628 | 0.5764 | 0.808 |
| Random Forest | 0.7607 | 0.5337 | 0.7466 | 0.6225 | 0.837 |
| **XGBoost** | 0.7229 | 0.4858 | **0.8302** | 0.6129 | **0.8297** |
| Rule-Based (Baseline) | 0.7393 | 0.6087 | 0.0377 | 0.0711 | N/A |

---

## Breaking Down Each Model

### 1. Logistic Regression
Honestly this performed better than I expected for a baseline. It got **76.55% Recall**
which means it caught about 3 out of 4 churning customers — not bad at all for the
simplest model in the lineup. The ROC-AUC of 0.832 was also surprisingly strong.

The weakness is precision (0.50) — it flags a lot of non-churners as churners too.
But for this problem, I'd rather have that than miss actual churners.

**Confusion matrix showed:** 754 correct non-churners, 284 correct churners,
87 missed churners, 279 false alarms.

---

### 2. Decision Tree
This one was the most interpretable — I could literally plot the tree and see the
exact questions it was asking. The top split was always around tenure and contract type,
which matched what I found in EDA.

Performance-wise it was the weakest of the four — 0.7037 accuracy and 0.808 AUC.
The recall (76.28%) was decent but the F1 (0.5764) suffered because precision dropped
to 0.46. Setting max_depth=6 helped with overfitting but there's only so much a single
tree can do.

**Best use case:** When you need to explain the model to someone non-technical.
You can literally show them the tree diagram.

---

### 3. Random Forest
This was the most **balanced** model overall. It had the highest accuracy (76.07%)
and the highest ROC-AUC (0.837) across all models. The F1 score of 0.6225 was also
the best of the four.

The trade-off was recall — at 74.66% it caught slightly fewer churners than Logistic
Regression or XGBoost. But it made up for it with much better precision (0.5337),
meaning fewer false alarms.

The feature importance chart from Random Forest was really useful — tenure,
TotalCharges, Income, and charges_per_service came out as the top predictors.
That confirmed what I suspected from the correlation heatmap.

**Confusion matrix showed:** 791 correct non-churners, 277 correct churners,
94 missed churners, 242 false alarms — the cleanest matrix of the four.

---

### 4. XGBoost
XGBoost had the **highest Recall of all models at 83.02%** — meaning it caught
the most actual churners. That matters most in this problem. It also had the
second-highest AUC at 0.8297.

The accuracy (72.29%) looks lower than Random Forest, but that's actually expected
when you tune for recall — you'll flag more non-churners too (higher false positives),
which pulls accuracy down while keeping recall high.

**Confusion matrix showed:** 707 correct non-churners, 308 correct churners,
63 missed churners, 326 false alarms. The 63 missed churners is the lowest of all models.

---

### Rule-Based System (Baseline)
This was a hand-coded set of if-else rules I built after EDA. The results were
eye-opening — it got 73.93% accuracy but a Recall of only **3.77%**.

That means it barely caught any churners at all. It was too conservative —
only flagging the most obvious cases (month-to-month + tenure < 12 months).
Real churn behaviour is more complex than a few rules can capture.

It did get the highest Precision (0.608) of anything — but precision without
recall is useless here. Missing 96% of churners is not a viable business strategy.

---

## Why Recall Is the Right Metric Here

I want to be clear about why I focused on Recall over Accuracy.

If the model misses a churning customer (False Negative) → that customer leaves
quietly and the company loses 100% of their future revenue.

If the model flags a non-churner (False Positive) → the company sends them a
retention offer or discount. Small cost, easily recovered.

So the cost of a False Negative is much higher than a False Positive.
That's why Recall matters more than raw accuracy in churn prediction.

---

## Final Recommendation

**For maximum churn detection → XGBoost** (Recall: 83%, AUC: 0.83)

**For balanced performance → Random Forest** (F1: 0.62, AUC: 0.837)

**For business explainability → Decision Tree** (can be shown as a diagram)

In a real production system, I'd deploy XGBoost for predictions and keep the
Decision Tree handy for explaining decisions to business stakeholders who
aren't comfortable with black-box models.

---

## What I'd Do Differently With More Time

- Run proper hyperparameter tuning with Optuna instead of manual trial-and-error
- Look into SHAP values to explain individual XGBoost predictions
- Try a stacking ensemble — combining all four models might outperform any single one
- Build a simple API endpoint so the model can actually serve predictions in real time

---

*Written by [Your Name] | Internship Assessment Submission*
