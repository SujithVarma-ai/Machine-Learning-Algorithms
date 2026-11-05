# 🤖 Machine Learning Algorithms from Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SujithVarma-ai/Machine-Learning-Algorithms/blob/main/notebooks/ML_Algorithms_from_scratch.ipynb)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![NumPy](https://img.shields.io/badge/NumPy-only-orange)

> 10 core Machine Learning algorithms, each explained in plain English, worked out by hand in my notes, and coded from scratch using only **Python + NumPy** (matplotlib for plots).

New to Machine Learning? You are in the right place. Every algorithm below has:
1. **What it does** in simple words
2. **A real-life example**
3. **The key formula**
4. **A worked example with the real numbers from my Colab notebook**

---

## 📑 Table of Contents
- [Algorithms at a Glance](#-algorithms-at-a-glance)
- [Results at a Glance](#-results-at-a-glance)
- [Two Ideas You Need First](#-two-ideas-you-need-first)
- [The 10 Algorithms](#-the-10-algorithms)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)
- [What I Learned](#-what-i-learned)
- [Roadmap](#-roadmap)
- [Author](#-author)

---

## 🗺️ Algorithms at a Glance

| # | Algorithm | Type | Used For | Notes | Code |
|---|-----------|------|----------|:----:|:----:|
| 1 | Linear Regression | Supervised | Predicting a number | ✅ | ✅ |
| 2 | Logistic Regression | Supervised | Yes/No decisions (fraud or not) | ✅ | ✅ |
| 3 | K-Nearest Neighbors | Supervised | Classify by looking at similar points | ✅ | ✅ |
| 4 | Decision Tree | Supervised | Decisions as a flowchart | ✅ | ✅ |
| 5 | Random Forest | Supervised (Ensemble) | Many trees voting together | ✅ | ✅ |
| 6 | Support Vector Machine | Supervised | Best line separating two classes | ✅ | ✅ |
| 7 | Naive Bayes | Supervised | Classification using probability | ✅ | ✅ |
| 8 | K-Means | Unsupervised | Grouping similar data | ✅ | ✅ |
| 9 | PCA | Unsupervised | Reducing the number of features | ✅ | ✅ |
| 10 | XGBoost (simplified) | Supervised (Ensemble) | Trees that fix each other's mistakes | ✅ | ✅ |

**Supervised Learning** - Learns from labelled training data, where each input has a known correct output (label). It uses these examples to make predictions on new data.

**Unsupervised Learning** - Learns from unlabelled data, where the correct outputs are not provided. It identifies hidden patterns, structures, or groups in the data.

---

## 📊 Results at a Glance

| Algorithm | Result from the notebook |
|---|---|
| Linear Regression | `w = 1.144`, `b = 1.189`, MSE `0.086`. Predicts `11.49` for X = 9 |
| Logistic Regression | Loss `0.049`. A ₹500 transaction gets probability `0.524` → class 1 |
| KNN (K=3) | Student with 4 hrs study and 7 hrs sleep → **Pass** |
| Decision Tree | Root = **Outlook** (information gain `0.247`) |
| Random Forest | 5 trees → 3 Yes, 2 No → **Yes** |
| SVM | Boundary near `x₁ = 3`, margin width `1.895` |
| Naive Bayes | Score `0.6` vs `0` → **Go Out** |
| K-Means | Converged after 4 passes, 3 clusters |
| PCA | 2 features → 1, **100%** variance kept |
| XGBoost | MSE `125 → 69.4 → 44.8 → 30.5` over 3 trees |

---

## 🧠 Two Ideas You Need First

### 1. Loss function: "How wrong is the model?"
| Loss | Formula | Use when |
|------|---------|----------|
| MAE | `(1/n) Σ │yᵢ − ŷᵢ│` | You want the average error and don't want big errors to dominate |
| MSE | `(1/n) Σ (yᵢ − ŷᵢ)²` | You want to punish large errors heavily |
| RMSE | `√MSE` | You want the error in the original units |

### 2. Gradient Descent: "How does the model improve?"
The model makes a prediction, measures its error, and adjusts its parameters a little. It repeats this many times.

```
Prediction → Loss → Gradient → Update parameters → Repeat

w_new = w_old − η · ∂L/∂w
b_new = b_old − η · ∂L/∂b       (η = learning rate)
```

---

## 📚 The 10 Algorithms

### 1. Linear Regression
**In simple words:** Draws the best straight line through the data to predict a number.
**Example:** Predict a house price from its size.

**Key formulas:**
```
ŷ = w·x + b
MSE = (1/n) Σ (ŷ − y)²
∂L/∂w = 2 · mean(error · x)      ∂L/∂b = 2 · mean(error)
```

**Worked example (notebook):** 8 points, `X = 1…8`, `y = 2.1, 3.9, 4.2, 6.1, 6.8, 8.3, 8.9, 10.4`.
Gradient descent with `η = 0.02` for `1000` epochs gives:

| Learned | Value |
|---|---|
| w | 1.144 |
| b | 1.189 |
| MSE | 0.086 |

Prediction for `X = 9` → `1.144 × 9 + 1.189 = 11.49` ✅

*In my notes, the same idea is shown with house prices: with `w = 0.03` and `b = 10`, a 2000 sq ft house costs `0.03 × 2000 + 10 = 70` lakhs.*

---

### 2. Logistic Regression
**In simple words:** Predicts the *probability* of something being Yes or No.
**Example:** Is this transaction fraud?

**Key formulas:**
```
z = w·x + b
p = σ(z) = 1 / (1 + e^(−z))        → always between 0 and 1
Loss (Binary Cross-Entropy) = −[ y·log(p) + (1−y)·log(1−p) ]
```
If `p > 0.5` → predict class 1 (Fraud), else class 0.

**Worked example (notebook):** Amounts `₹100, ₹300, ₹700, ₹900` with labels `0, 0, 1, 1`.
Amounts are scaled by dividing by 1000 (otherwise training is unstable). Trained with `η = 0.1` for `5000` epochs:

| Learned | Value |
|---|---|
| w | 11.90 |
| b | −5.85 |
| Final loss | 0.049 |

| Amount | Predicted probability |
|---|---|
| ₹100 | 0.01 |
| ₹300 | 0.09 |
| ₹700 | 0.92 |
| ₹900 | 0.99 |

A new ₹500 transaction gets `p = 0.524` → class 1 ✅ (it sits right next to the decision boundary).

*In my notes, I traced one gradient step by hand: starting from `w = 0, b = 0`, both predictions are `0.5` and the loss is `0.693`.*

---

### 3. K-Nearest Neighbors (KNN)
**In simple words:** To classify a new point, look at its K closest neighbours and let them vote.
**Example:** Will a student who studied 4 hrs and slept 7 hrs pass?

**Key formula:** `d = √( (x₁−a₁)² + (x₂−a₂)² )`

**Worked example (notebook, K = 3):**

| Student | Study hrs | Sleep hrs | Result | Distance to (4, 7) |
|---|---|---|---|---|
| A | 1 | 7 | Fail | 3.000 |
| B | 2 | 6 | Fail | 2.236 |
| C | 3 | 7 | Pass | **1.000** |
| D | 6 | 8 | Pass | **2.236** |
| E | 7 | 6 | Pass | 3.162 |

Nearest 3 = C, B, D with labels `Pass, Fail, Pass` → 2 Pass vs 1 Fail → **Prediction: Pass** ✅

---

### 4. Decision Tree
**In simple words:** Asks a series of yes/no questions, like a flowchart, to reach a decision.
**Example:** Should we play tennis today?

**Key formulas:**
```
Entropy: E = − Σ pᵢ · log₂(pᵢ)          (how mixed the data is)
Information Gain = Entropy(before) − weighted Entropy(after split)
```

**Worked example (notebook):** 14 days of "Play Tennis". Parent entropy = `0.94`.

| Feature | Information Gain |
|---|---|
| **Outlook** | **0.247** ← highest, becomes the root |
| Humidity | 0.152 |
| Windy | 0.048 |
| Temperature | 0.029 |

The tree is built recursively (ID3):
```
Outlook
├── Sunny    → Humidity (High → No, Normal → Yes)
├── Overcast → Yes
└── Rainy    → Windy    (Weak → Yes, Strong → No)
```

---

### 5. Random Forest
**In simple words:** Many decision trees, each trained a little differently, vote on the answer. One tree can memorise the data (overfitting); a crowd is more reliable.

**Two sources of randomness:**
1. **Bootstrap sampling:** each tree trains on a random sample of the data
2. **Random feature selection:** each split considers only a random subset of features

**Worked example (notebook):** 5 trees, each using at most 2 random features per split, trained on the Play Tennis data. New day: `Sunny, Cool, High, Strong`.

| Tree | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Vote | Yes | No | Yes | Yes | No |

3 Yes vs 2 No → **Final: Yes** (majority voting) ✅

> 📝 To get a clear 3–2 vote for this demo, the code tries random seeds until one gives it (seed `3`). With other seeds the vote can differ.

---

### 6. Support Vector Machine (SVM)
**In simple words:** Finds the line that separates two classes with the **widest possible gap** (margin). Only the points closest to the line, the **support vectors**, matter.
**Example:** Separating healthy cells from cancerous cells.

**Key formulas:**
```
f(x) = wᵀx + b        → predict +1 if f(x) ≥ 0, else −1
Minimise ½‖w‖²   subject to   yᵢ(wᵀxᵢ + b) ≥ 1
Margin width = 2 / ‖w‖
```

**Worked example (notebook):**

| Point | x₁ | x₂ | Class |
|---|---|---|---|
| A | 2 | 2 | +1 |
| B | 2 | 4 | +1 |
| C | 4 | 2 | −1 |
| D | 4 | 4 | −1 |

Trained with hinge-loss gradient descent (`η = 0.01`, `λ = 0.01`, `5000` epochs):

| Learned | Value |
|---|---|
| w | (−1.056, 0.0007) |
| b | 3.16 |
| Decision scores | 1.05, 1.051, −1.061, −1.06 |
| Predictions | +1, +1, −1, −1 (all correct ✅) |
| Margin width | 1.895 |

The exact hard-margin solution is `w = (−1, 0)`, `b = 3` (boundary at `x₁ = 3`, margin `2`). Gradient descent gets very close to it.

*SVMs can also draw curved boundaries using the **kernel trick**.*

---

### 7. Naive Bayes
**In simple words:** Uses probability. For each class, multiply how likely each observed feature is, then pick the class with the highest score.
**Example:** Will I go out or stay home, given sunny weather and a working car?

**Key formula:** `Score(C) = P(C) · Π P(xᵢ | C)`

**Worked example (notebook):**

| Probability | Go Out | Stay Home |
|---|---|---|
| Prior `P(class)` | 0.60 | 0.40 |
| `P(Sunny │ class)` | 1.00 | 0.00 |
| `P(Working │ class)` | 1.00 | 0.00 |
| **Score** | **0.60** | **0.00** |

New observation `Sunny, Working` → **Prediction: Go Out** ✅

⚠️ **Watch out:** one zero probability wipes out the whole score. The usual fix is **Laplace smoothing** (add 1 to every count). It is not used in this small example.

---

### 8. K-Means Clustering
**In simple words:** Groups unlabeled data into K clusters. Repeat: assign each point to the nearest centre, then move each centre to the average of its points.

**Steps:**
```
1. Pick K starting centroids
2. Assign every point to its nearest centroid
3. Recompute each centroid as the mean of its points
4. Repeat 2–3 until the centroids stop changing
```

**Worked example (notebook, K = 3):** 8 points: A1 (2,10), A2 (2,5), A3 (8,4), B1 (5,8), B2 (7,5), B3 (6,4), C1 (1,2), C2 (4,9).
Starting centroids: `(2,10)`, `(5,8)`, `(1,2)`.

| Pass | Centroid 1 | Centroid 2 | Centroid 3 |
|---|---|---|---|
| 1 | (2, 10) | (6, 6) | (1.5, 3.5) |
| 2 | (3, 9.5) | (6.5, 5.25) | (1.5, 3.5) |
| 3 | (3.67, 9) | (7, 4.33) | (1.5, 3.5) |
| 4 | (3.67, 9) | (7, 4.33) | (1.5, 3.5) ✅ no change |

**Final clusters:**
- Cluster 1: A1, B1, C2
- Cluster 2: A3, B2, B3
- Cluster 3: A2, C1

---

### 9. Principal Component Analysis (PCA)
**In simple words:** Squeezes many features into fewer ones while keeping as much information (variance) as possible.
**Example:** Combine Math and Science scores into a single "overall ability" score.

**Steps:**
```
1. Centre the data (subtract the mean)
2. Compute the covariance matrix
3. Find eigenvalues and eigenvectors
4. Keep the eigenvector with the largest eigenvalue (PC1)
5. Project the data onto it
```

**Worked example (notebook):**

| Student | Math | Science |
|---|---|---|
| A | 2 | 1 |
| B | 3 | 2 |
| C | 4 | 3 |
| D | 5 | 4 |

- Mean = `[3.5, 2.5]`
- Covariance matrix = `[[1.667, 1.667], [1.667, 1.667]]`
- Eigenvalues = `3.333` and `0`
- PC1 = `[0.7071, 0.7071]`
- PC1 values for A, B, C, D = `−2.1213, −0.7071, 0.7071, 2.1213`
- Explained variance = `3.333 / (3.333 + 0) = 100%`

The two features were perfectly correlated, so **nothing was lost** ✅

---

### 10. XGBoost (Extreme Gradient Boosting): simplified
**In simple words:** Builds trees **one after another**, and each new tree fixes the mistakes of the model so far.

| | Random Forest | XGBoost |
|---|---|---|
| Trees built | Independently | Sequentially |
| Each tree | Votes | Corrects previous errors |
| Final answer | Voting / averaging | Sum of all corrections |

**Key formulas:**
```
New prediction = Old prediction + η × Tree output
Leaf value     = − G / (H + λ)         (G = sum of gradients, H = sum of Hessians)
Split gain     = ½ [ G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ) ]
```

**Worked example (notebook):** Study hours `1, 2, 3, 4` → scores `30, 40, 50, 60`.
Settings: `η = 0.5`, `3 trees`, `max_depth = 1`, `λ = 1`.
Starting prediction = mean of scores = `45`.

| Stage | Predictions | MSE |
|---|---|---|
| Start | 45, 45, 45, 45 | 125 |
| After tree 1 | 41.667, 41.667, 48.333, 48.333 | 69.444 |
| After tree 2 | 39.444, 39.444, 50.556, 50.556 | 44.753 |
| After tree 3 | 37.083, 40.625, 51.736, 51.736 | 30.468 |

Each new tree improves the model, and the predictions move toward `30, 40, 50, 60` ✅

> This is a small educational version (gradient, Hessian, L2 regularisation and split gain), not the full XGBoost library.

---

## 📁 Repository Structure

```text
Machine-Learning-Algorithms/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── ML_Algorithms_from_scratch.ipynb
│
├── notes/
│   ├── 01_Linear_Regression.md
│   ├── 02_Logistic_Regression.md
│   ├── 03_KNN.md
│   ├── 04_Decision_Tree.md
│   ├── 05_Random_Forest.md
│   ├── 06_SVM.md
│   ├── 07_Naive_Bayes.md
│   ├── 08_KMeans.md
│   ├── 09_PCA.md
│   └── 10_XGBoost.md
│
└── images/            ← plots and handwritten-note photos
```

---

## ▶️ How to Run

**Option 1: No setup (recommended).** Click the **Open in Colab** badge at the top.

**Option 2: On your computer**
```bash
git clone https://github.com/SujithVarma-ai/Machine-Learning-Algorithms.git
cd Machine-Learning-Algorithms
pip install -r requirements.txt
jupyter notebook notebooks/ML_Algorithms_from_scratch.ipynb
```

`requirements.txt`:
```
numpy
matplotlib
```

---

## 💡 What I Learned

- Almost every model follows the same loop: **predict → measure error → adjust**.
- Many "different" algorithms share ideas: Random Forest and XGBoost are both built from Decision Trees.
- Doing the maths by hand first made the NumPy code much easier to write.
- Small details matter: scaling features (Logistic Regression), smoothing zero probabilities (Naive Bayes), choosing K (KNN, K-Means).

---

## 🛣️ Roadmap

- [ ] Ridge Regression
- [ ] Lasso Regression
- [ ] Perceptron
- [ ] t-SNE
- [ ] Neural networks and Deep Learning

---

## 🙋 Author

**Sujith Varma**
GitHub: [SujithVarma-ai](https://github.com/SujithVarma-ai)
Focus: Machine Learning · Artificial Intelligence · Deep Learning · Large Language Models
