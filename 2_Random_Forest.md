# 2. Random Forest 실습

## 1. Import dataset


```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
```

## 2. Split dataset


```python
dataset = load_breast_cancer().data
target = load_breast_cancer().target
features = load_breast_cancer().feature_names

X_train, X_test, y_train, y_test = train_test_split(dataset, target, test_size=0.2)
```

## 3. Import random forest


```python
from sklearn.ensemble import RandomForestClassifier
```

## Official Documentation
(https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

- n_estimators : int, default=100
     - The number of trees in the forest.



- criterion : {“gini”, “entropy”, “log_loss”}, default=”gini”
    - The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain, see Mathematical formulation. Note: This parameter is tree-specific.

- max_depth : int, default=None
    - The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

- min_samples_split : int or float, default=2
    - The minimum number of samples required to split an internal node:

    - If int, then consider min_samples_split as the minimum number.

    - If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.

- min_samples_leaf : int or float, default=1
    - The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.

    - If int, then consider min_samples_leaf as the minimum number.

    - If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.

- min_weight_fraction_leaf : float, default=0.0
    - The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.

- max_features : {“sqrt”, “log2”, None}, int or float, default=”sqrt”
    - The number of features to consider when looking for the best split:

    - If int, then consider max_features features at each split.

    - If float, then max_features is a fraction and max(1, int(max_features * n_features_in_)) features are considered at each split.

    - If “auto”, then max_features=sqrt(n_features).

    - If “sqrt”, then max_features=sqrt(n_features).

    - If “log2”, then max_features=log2(n_features).

    - If None, then max_features=n_features.

- max_leaf_nodes : int, default=None
    - Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.

- min_impurity_decrease : float, default=0.0
    - A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

    - The weighted impurity decrease equation is the following:

    - N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)
    - where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of samples in the left child, and N_t_R is the number of samples in the right child.

    - N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed.

- bootstrap : bool, default=True
    - Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.

- oob_score : bool, default=False
    - Whether to use out-of-bag samples to estimate the generalization score. Only available if bootstrap=True.

- n_jobs : int, default=None
    - The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized over the trees. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.

- random_state : int, RandomState instance or None, default=None
    - Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features). See Glossary for details.

- verbose : int, default=0
    - Controls the verbosity when fitting and predicting.

- warm_start : bool, default=False
    - When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest. See the Glossary.

- class_weight : {“balanced”, “balanced_subsample”}, dict or list of dicts, default=None
    - Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.

    - Note that for multioutput (including multilabel) weights should be defined for each class of every column in its own dict. For example, for four-class multilabel classification weights should be [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of [{1:1}, {2:5}, {3:1}, {4:1}].

    - The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))

    - The “balanced_subsample” mode is the same as “balanced” except that weights are computed based on the bootstrap sample for every tree grown.

    - For multi-output, the weights of each column of y will be multiplied.

    - Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.

- ccp_alphan : on-negative float, default=0.0
    - Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed. See Minimal Cost-Complexity Pruning for details.

- max_samples : int or float, default=None
    - If bootstrap is True, the number of samples to draw from X to train each base estimator.

    - If None (default), then draw X.shape[0] samples.

    - If int, then draw max_samples samples.

    - If float, then draw max_samples * X.shape[0] samples. Thus, max_samples should be in the interval (0.0, 1.0].

## Random Forest Classifier 호출


```python
rf = RandomForestClassifier(random_state=42)
```

## Random Forest Classifier 학습


```python
rf.fit(X_train, y_train)
```

## 학습된 모델로 Test set 예측


```python
pred = rf.predict(X_test) # Binary classification
pred_proba = rf.predict_proba(X_test) # Probability classification
```


```python
print(pred)
```


```python
print(pred_proba[:10])
```

## Metric 구하기


```python
def metrics(pred, answ, thr=0.5):
    """
    Function that prints overall metrics
    - Input variables
        - pred : Prediction value. int(0 or 1) or float(0 ~ 1).
        - answ : True label. One of 0 and 1.
        - thr  : threshold for classification
    
    - tp : True Positive
    - fp : False Positive (Model output > threshold, True label < threshold)
    - fn : False Negative (Model output < threshold, True label > threshold)
    - tn : True Negative
    
    - Print values
        - Accuracy             : (tp + tn) / Total
        - Sensitivity (recall) : tp / (tp + fn) 
        - Specificity          : tn / (tn + fp)
        - Precision            : tp / (tp + fp)
        - F1 score             : 2 * (sensitivity * precision) / (sensitivity + precision)
    """
    tp, fp, fn, tn = 0, 0, 0, 0
    assert len(pred) == len(answ), "Length of prediction should be same as length of answer. " + \
                                    f"Check length of prediction (current : {len(pred)}) and " + \
                                    f"answer (current : {len(answ)})."
    for i in range(len(pred)):
        p = pred[i]
        assert 0 <= p <= 1, "Prediction value should range from 0 to 1"
        if p>=0.5 and answ[i]>=0.5:
            tp += 1
        elif p<0.5 and answ[i]>=0.5:
            fn += 1
        elif p>=0.5 and answ[i]<0.5:
            fp += 1
        else:
            tn += 1
    assert len(pred) == tp+fn+fp+tn
    acc = (tp + tn) / len(pred) # Accuracy
    sen = tp / (tp + fn) # Sensitivity (= Recall)
    spe = tn / (tn + fp) # Specificity
    pre = tp / (tp + fp) # Precision (= Positivie Predictive Value; PPV)
    f1 = 2/(1/sen + 1/pre)
    print("Overall metrics")
    print(f"Accuracy   : {acc:.3f}")
    print(f"Recall     : {sen:.3f}")
    print(f"Specifcity : {spe:.3f}")
    print(f"Precision  : {pre:.3f}")
    print(f"F1 Score   : {f1:.3f}")
```


```python
metrics(pred, y_test)
```


```python
rf.score(X_test, y_test) # Overall Accuracy. This should be same as Accuracy above.
```

## Feature Importance를 통해 어떤 feature이 중요하게 작용하는지 시각화


```python
fi = rf.feature_importances_
```


```python
import matplotlib.pyplot as plt
```


```python
plt.figure(figsize=(10,10))
plt.bar(features, fi)
plt.xticks(rotation=90)
plt.show()
```


```python

```
