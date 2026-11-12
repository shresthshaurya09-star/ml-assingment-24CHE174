# Cancer Classification: Logistic Regression vs Random Forest

## Summary

- **Dataset:** Cancer Wisconsin (Diagnostic) from Kaggle. It has 569 tumour samples, each described by 30 measurements of the cell nuclei (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension). For each one, the mean, the standard error and the "worst" (largest) value are recorded. 357 samples are benign and 212 are malignant.
- **What I'm predicting:** whether a tumour is malignant (1) or benign (0). This is a binary classification problem.
- **Models:** Logistic Regression (on standardised features) and Random Forest with 200 trees.
- **Result:** both reach **96.5% accuracy** on the held-out test set (110 of 114 correct). With 5-fold cross-validation, Logistic Regression averages 97.4% and Random Forest 95.4%.
- **One thing I learned:** accuracy hides what kind of mistakes a model makes. For cancer, missing a malignant tumour is worse than a false alarm, so recall on the malignant class matters as much as accuracy.

The rest of this file explains the maths behind each step and shows the actual numbers from my run.

---

## 1. Data preparation

- I dropped the `id` column because it is just a row label.
- There were no missing values and no duplicate rows, so no imputation was needed.
- I encoded the target as M = 1 and B = 0.
- I split the data 80/20 with `random_state=42` and `stratify=y`, giving 455 training and 114 test samples. Stratifying keeps the class mix almost the same in both parts: 37.4% malignant in training (170 of 455) and 36.8% in the test set (42 of 114).

### Standardisation (z-score)

The features live on very different scales (area is in the hundreds, smoothness is around 0.1). Logistic Regression is sensitive to that, so each feature $x$ is rescaled using the training set's mean $\mu$ and standard deviation $\sigma$:

$$z = \frac{x - \mu}{\sigma}$$

Example from my data: for `radius_mean` the training mean is $\mu = 14.166$ and $\sigma = 3.575$. The first test sample has a radius of 11.41, so

$$z = \frac{11.41 - 14.166}{3.575} = -0.771$$

meaning it is about 0.77 standard deviations below average. I fitted the scaler on the training data only and then applied the same $\mu$ and $\sigma$ to the test data, so no information from the test set leaks into training. Random Forest only compares values to thresholds, so it doesn't need scaling.

---

## 2. Model 1: Logistic Regression

### The model

For a sample with feature vector $\mathbf{x}$, weights $\mathbf{w}$ and intercept $b$, the model first computes a linear score (the log-odds):

$$z = \mathbf{w}^\top \mathbf{x} + b$$

and turns it into a probability with the sigmoid function:

$$p = P(y=1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

The sigmoid squashes any number into the range 0 to 1. The prediction is malignant when $p \ge 0.5$, which happens exactly when $z \ge 0$. So the decision boundary is the flat surface $\mathbf{w}^\top \mathbf{x} + b = 0$ in the 30-dimensional feature space.

Worked example: for the first test sample, my trained model gives $z = -7.92$, so

$$p = \frac{1}{1 + e^{7.92}} \approx 0.00036$$

That is a 0.036% chance of malignant, so it predicts benign. The true label is benign, so it is correct.

### How the weights are learned

Training finds the $\mathbf{w}$ and $b$ that minimise the **log-loss** (binary cross-entropy) over the $n$ training samples:

$$J(\mathbf{w}, b) = -\frac{1}{n} \sum_{i=1}^{n} \Big[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \Big]$$

If the true label is 1, only $\log p_i$ counts, so the loss is large when the model gives a small probability. If the true label is 0, only $\log(1 - p_i)$ counts. This punishes confident wrong answers very heavily. The gradient has a neat form:

$$\frac{\partial J}{\partial w_j} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)\, x_{ij}$$

and the solver (lbfgs in scikit-learn) follows it downhill until the loss stops improving. scikit-learn also adds an L2 penalty $\tfrac{1}{2C}\lVert \mathbf{w} \rVert^2$ with the default $C = 1$, which stops the weights from growing too large and overfitting.

### What the weights mean

Because the features are standardised, each coefficient $w_j$ is the change in log-odds of malignancy for a one-standard-deviation increase in feature $j$, with the other features held fixed. Increasing a feature by one standard deviation multiplies the odds by $e^{w_j}$. The largest coefficients in my model were `texture_worst` (1.43), `radius_se` (1.23), `symmetry_worst` (1.06), `concave points_mean` (0.95) and `concavity_worst` (0.91). The intercept was $b = -0.243$.

---

## 3. Model 2: Random Forest

A Random Forest is many decision trees whose votes are combined.

### Gini impurity (how a tree chooses a split)

A node containing a fraction $p_k$ of class $k$ has Gini impurity

$$G = 1 - \sum_{k} p_k^2$$

It is 0 when the node is pure and 0.5 at most for two classes. For my whole dataset, $p_{malignant} = 212/569 = 0.3726$, so

$$G = 1 - (0.3726^2 + 0.6274^2) = 1 - (0.1388 + 0.3936) = 0.4675$$

A tree tries splits of the form "is feature $j \le t$?" and picks the one that gives the biggest drop in impurity (the information gain):

$$\Delta G = G_{parent} - \left( \frac{n_{left}}{n} G_{left} + \frac{n_{right}}{n} G_{right} \right)$$

It then repeats this on each child until the leaves are pure or a stopping rule is reached.

### Why a forest is better than one tree

- **Bagging:** each of the 200 trees is trained on a bootstrap sample, which means $n$ rows drawn with replacement. About $1 - (1 - 1/n)^n \to 1 - 1/e \approx 63.2\%$ of the distinct rows appear in any one sample.
- **Random feature subsets:** at each split only $\sqrt{30} \approx 5$ randomly chosen features are considered. This makes the trees different from each other.
- **Voting:** the forest averages the trees' predicted class probabilities and picks the class with the higher average, which behaves like a majority vote. Averaging many different trees lowers the variance, so the forest overfits much less than a single deep tree.

### Feature importance

A feature's importance is the total impurity decrease it produces across all splits in all trees (weighted by how many samples reach each split), then normalised to sum to 1. My top features were `perimeter_worst` (0.148), `area_worst` (0.132), `concave points_worst` (0.110), `concave points_mean` (0.088) and `radius_worst` (0.085).

---

## 4. Evaluation metrics

From the confusion matrix, with malignant as the positive class: TP = malignant correctly found, TN = benign correctly found, FP = benign wrongly called malignant, FN = malignant wrongly called benign.

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \qquad \text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall} = \frac{TP}{TP + FN}$$

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Results on the test set (114 samples)

| | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 71 | 1 | 3 | 39 |
| Random Forest | 72 | 0 | 4 | 38 |

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | (71 + 39) / 114 = 0.9649 | (72 + 38) / 114 = 0.9649 |
| Precision (malignant) | 39 / 40 = 0.975 | 38 / 38 = 1.000 |
| Recall (malignant) | 39 / 42 = 0.929 | 38 / 42 = 0.905 |
| F1 (malignant) | 0.951 | 0.950 |
| Log-loss | 0.077 | 0.115 |

Both models get 110 out of 114 right, but they fail differently. Random Forest never raises a false alarm (precision 1.0) but misses one more malignant tumour than Logistic Regression. Since a missed malignant tumour is the more serious mistake, Logistic Regression's slightly higher recall is the better trade-off here. Its lower log-loss also means its probabilities are better calibrated.

### Cross-validation

A test set of 114 rows is small, so one or two samples change the score by about 1 percentage point. I also ran stratified 5-fold cross-validation, which trains and tests on five different splits and averages the accuracy:

| Model | Fold accuracies | Mean | Std |
|---|---|---|---|
| Logistic Regression | 0.974, 0.947, 0.965, 0.991, 0.991 | 0.974 | 0.017 |
| Random Forest | 0.965, 0.939, 0.956, 0.947, 0.965 | 0.954 | 0.010 |

Averaged over several splits, Logistic Regression comes out slightly ahead. With data this small and clean, with strong linear structure, a simple model can match or beat a more complex one.

---

## 5. Observations and conclusions

### Observations from the data

- **Class balance:** 357 samples (62.7%) are benign and 212 (37.3%) are malignant. This is uneven but not extreme, so accuracy is a fair starting metric, but it needs precision and recall beside it. A model that always said "benign" would already score 62.7%, so that is the number to beat.
- **Cleanliness:** there were no missing values and no duplicate rows, so cleaning was limited to dropping `id`, encoding the target and scaling.
- **Malignant tumours are larger and more irregular:** the average `radius_mean` is 17.46 for malignant against 12.15 for benign, the average `area_mean` is 978 against 463, and the average `concave points_mean` is 0.088 against 0.026. Texture also differs (21.6 against 17.9), though less sharply.
- **Many features are redundant:** `radius_mean` and `perimeter_mean` have a correlation of 0.998, and `radius_mean` and `area_mean` 0.987. The same pattern holds for the "worst" columns. Across all 30 features, 21 pairs have an absolute correlation above 0.9, so the dataset carries much less independent information than 30 columns suggest.

### Observations from the models

- **Same accuracy, different mistakes:** both models score 96.5% (110 of 114) on the test set. Logistic Regression makes 1 false alarm and 3 missed malignant cases. Random Forest makes 0 false alarms and 4 missed malignant cases. Random Forest is "safer" about false alarms (precision 1.0), while Logistic Regression catches more cancers (recall 0.929 against 0.905).
- **Probabilities:** Logistic Regression has the lower log-loss (0.077 against 0.115), so its confidence levels are better calibrated. That matters if the probability itself is going to be used rather than just the yes/no label.
- **Cross-validation changes the picture slightly:** on one 80/20 split the models tie, but averaged over 5 folds Logistic Regression scores 97.4% (std 0.017) and Random Forest 95.4% (std 0.010). Logistic Regression is ahead on average, though its scores vary more between folds, from 94.7% to 99.1%.
- **The two models rank features differently:** Random Forest puts `perimeter_worst`, `area_worst` and `concave points_worst` on top, which are mostly size and shape measurements. Logistic Regression gives its largest weights to `texture_worst`, `radius_se` and `symmetry_worst`. The correlated features explain this: when several features carry the same information, a linear model can spread the weight among them in many equivalent ways, and a forest can use any of them in different trees.

### Conclusions

1. For this dataset, both models separate benign from malignant tumours very well. Logistic Regression is at least as good as Random Forest, so the simpler and more interpretable model is the better choice here.
2. Accuracy alone is not enough for a medical problem. A false negative (a missed malignant tumour) is the costliest error, so recall on the malignant class should guide the final choice of model and threshold. Lowering the decision threshold below 0.5 would trade some extra false alarms for fewer missed cases.
3. Because the dataset is small (114 test samples), one or two samples move the test accuracy by about 1 percentage point. Cross-validation is a more trustworthy comparison than a single split, and the difference between the two models is small enough that I would not call either one clearly better.
4. Feature importance should be read with care when features are correlated. The importance of one feature does not mean the others are unimportant, since they carry nearly the same information.

### Limitations

- The data comes from one source with only 569 samples, so the results may not carry over to other hospitals or imaging equipment.
- I used default hyperparameters for both models and did no tuning, so both could probably improve slightly.
- These are features extracted from images of cells, not a clinical diagnosis. A model like this could only support a doctor, never replace one.

## Files

- `cancer_classification.ipynb`: the full notebook (loading, cleaning, both models, evaluation)
- `cancer.csv`: the dataset from Kaggle
- `README.md`: this file
