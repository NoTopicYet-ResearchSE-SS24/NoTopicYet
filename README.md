# HeartPredict

<img src="logo/logo.png" width="300" alt="logo">

HeartPredict is a Python library designed to analyze
and predict heart failure outcomes using patient data.

## Dataset information

The dataset used for this analysis was obtained from kaggle.com.
It contains 5000 medical records of patients who had heart-failure
and is licensed under CC0; made available under [this URL](https://www.kaggle.com/datasets/aadarshvelu/heart-failure-prediction-clinical-recordsselect=heart_failure_clinical_records.csv).

## Key Questions to Answer with the Dataset

### Descriptive Analysis

- What are the basic statistics (mean, median, standard deviation)
  of the clinical features?
- How is the age distribution of patients?
- What is the proportion of patients with conditions like anaemia, diabetes
  and high blood pressure?

### Correlation and Feature Importance

- Which clinical features are most strongly correlated with the DEATH_EVENT?
  And what are the most important features for predicting heart failure outcomes?
- How do different clinical features contribute
  to the risk of death due to heart failure?

### Predictive Analysis

- How accurately can we predict DEATH_EVENT using clinical features?
- Which machine learning model performs best for this prediction task?
  (We can use DecisionTrees, RandomForest or other classifiers) Let us use scikit-learn
- How do different models compare in terms of accuracy, precision, recall
  and other metrics?

### Survival Analysis

- What is the survival rate of patients over the follow-up period?
- How do survival rates vary with different clinical features
  (e.g., age, ejection fraction)?
- Can we identify patient subgroups with higher or lower survival probabilities?

### Risk Factor Analysis

- How does smoking affect the risk of death in heart failure patients?
- What is the impact of serum creatinine and serum sodium levels on patient outcomes?
- How does the combination of multiple risk factors affect the likelihood
  of heart failure-related death?

## Contributing

We welcome contributions from the community!
If you're interested in contributing to HeartPredict,
please take a look at our [CONTRIBUTING.md](CONTRIBUTING.md) file.
It contains all the guidelines you need to follow to get started,
including how to report issues, suggest features, and submit code.

## Code of Conduct

We are committed to providing a friendly, safe
and welcoming environment for everyone.
Please read our [Code of Conduct](CONDUCT.md)
to understand the standards we expect all members of our community to adhere to.

## License

This project is licensed under the MIT License -
see the [LICENSE](LICENSE) file for details.
