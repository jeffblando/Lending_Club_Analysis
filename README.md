# Credit Risk Analysis
Module 17 Challenge Files
- [credit_risk_resampling.ipynb](credit_risk_resampling.ipynb)
- [credit_risk_ensemble.ipynb](credit_risk_ensemble.ipynb)
- [Screenshots](Screenshots)

## Overview of Analysis
LendingClub, a peer to peer lending services company, is exploring usage of machine learning to predict credit risk. This analysis builds and evaluates several machine learning models/algorithms to predict credit risk ("high_risk" vs. "low_risk"). The models tested are:
- Naive Random Oversampling (with _RandomOverSampler_ algorithm)
- SMOTE Oversampling (with _SMOTE_ algorithm)
- Undersampling (with _ClusterCentroids_ algorithm
- Combination (Over and Under) Sampling (with _SMOTEENN_ algorithm)
- 2 additional machine learning models that reduce bias
  - Balanced Random Forest Classifier (with _BalancedRandomForestClassifier_)
  - Easy Ensemble AdaBoost Classifier (with _EasyEnsembleClassifier_)

## Results

- Naive Random Oversampling
  - Balanced Accuracy Score: ~0.65
    ![Screenshot](Screenshots/Screenshot_RandomOversampling%20AccuracyScore.png)
  - Precision: "high_risk" = 0.01, "low_risk" = 1.00
  - Recall: "high_risk" = 0.74, "low_risk" = 0.56
    ![Screenshot](Screenshots/Screenshot_RandomOversampling%20ClassificationReport.png)
- SMOTE Oversampling
  - Balanced Accuracy Score: ~0.65
    ![Screenshot](Screenshots/Screenshot_SMOTE%20AccuracyScore.png)
  - Precision: "high_risk" = 0.01, "low_risk" = 1.00
  - Recall: "high_risk" = 0.62, "low_risk" = 0.69
    ![Screenshot](Screenshots/Screenshot_SMOTE%20ClassificationReport.png)
- Undersampling
  - Balanced Accuracy Score: ~0.54
    ![Screenshot](Screenshots/Screenshot_UnderSampling%20AccuracyScore.png)
  - Precision: "high_risk" = 0.01, "low_risk" = 1.00
  - Recall: "high_risk" = 0.69, "low_risk" = 0.40
    ![Screenshot](Screenshots/Screenshot_UnderSampling%20ClassificationReport.png)
- SMOTEENN (Combination) Sampling
  - Balanced Accuracy Score: ~0.68
    ![Screenshot](Screenshots/Screenshot_SMOTEENN%20AccuracyScore.png)
  - Precision: "high_risk" = 0.01, "low_risk" = 1.00
  - Recall: "high_risk" = 0.80, "low_risk" = 0.56
    ![Screenshot](Screenshots/Screenshot_SMOTEENN%20ClassificationReport.png)
- Balanced Random Forest Classifier
  - Balanced Accuracy Score: ~0.79
    ![Screenshot](Screenshots/Screenshot_RandomForest%20AccuracyScore.png)
  - Precision: "high_risk" = 0.03, "low_risk" = 1.00
  - Recall: "high_risk" = 0.70, "low_risk" = 0.87
    ![Screenshot](Screenshots/Screenshot_RandomForest%20ClassificationReport.png)
- Easy Ensemble AdaBoost Classifier
  - Balanced Accuracy Score: ~0.93
    ![Screenshot](Screenshots/Screenshot_EasyEnsemble%20AccuracyScore.png)
  - Precision: "high_risk" = 0.09, "low_risk" = 1.00
  - Recall: "high_risk" = 0.92, "low_risk" = 0.94
    ![Screenshot](Screenshots/Screenshot_EasyEnsemble%20ClassificationReport.png)


## Summary
Given the disparate number of "low_risk" vs. "high_risk" loans, some adjustment is needed to accommodate this class imbalance. First, two oversampling methods were used: Random Oversampling and SMOTE Oversampling. Both produced accuracy scores ~0.65 and similar precision scores, but recall was similar between "high_risk" and "low_risk" for the SMOTE Oversampling, whereas there was a wider gap in recall between the two loan types for the Random Oversampling. Next, Undersampling was used, but it resulted in lower accuracy - with a balanced accuracy score of 0.54. Then, the combination method using SMOTEENN was run, and this performed the best out of the resampling methods used so far, with an accuracy score of 0.68. Across all of these similar precision was observed.

However, we also tried a couple ensemble classifiers, as these are models that also help to reduce bias. First, using a RandomForest Classifier, much stronger results were observed - an accuracy score of 0.79 and stronger precision for "high_risk" loans and stronge recall than the aforementioned resampling methods. 

Finally, the last model tested was the **Easy Ensemble AdaBoost Classifier** and this produced the strongest accuracy of ~0.93, stronger precision and recall than any of the other models tested. 

### Recommendation
Given these results, the Easy Ensemble AdaBoost Classifier seems like the best method to move forward with. However, there are a lot of features used in these models - it is likely worth narrowing down the list to exclude the lower ranked features to see if we can improve the model further. 
