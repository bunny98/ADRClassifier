# ADRClassifier

The file FullDataset.txt contains annotated tweets in the following form:

<Tweet ID>\t<User ID>\t<Class>\t<Tweet>

Class Label: 
0 for NoADR
1 for ADR

An SVM Classifier was fit on the above data after preprocessing, giving a F1-score of 0.5.
