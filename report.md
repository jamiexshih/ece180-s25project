Jamie Shih
06/13/2025
LeNet-5 CNN
STL-10

1. Training/validation loss plots
    Main Experiment: training loss shows steady downward trend in loss, flattening after ~70 epochs.
                     Validation loss plot shows validation accuracy improves up to ~74%, plateauing around epoch 80.

    Variation 1 is Batch Normalization
         training loss decreased faster in early epochs and validation accuracy improved more steadily; less overfitting seen.
         it reaches ~76.4% test accuracy.

   Variation 2 is L2 Regularization
        Loss curve similar to main experiment but slightly smoother.
        Validation accuracy plateaued earlier; test accuracy at ~75.6%.

    Variation 3 is Dropout
        Training loss decreases more slowly.
        Validation accuracy gradually improves, reaching the highest: ~76.8%.
        Dropout increases robustness and generalization.


2. Confusion Matrix & per-class accuracy
 Confusion matrix shows confusion between some similar classes.
Per-class accuracy:

Class	Accuracy (%)
0	75.2
1	71.6
2	78.4
3	66.0
4	79.8
5	72.6
6	70.0
7	77.2
8	76.0
9	73.6

3. Failed Examples with Analysis
Some misclassifications can be Class 3 and Class 5 due to their visual similarity in 32×32 resolution or Class 1 and Class 6 due to shape overlap in animals.

A lot of causes can be attributed to this, including:
  Loss of fine detail during downscaling to 32×32.
  Low inter-class distinction in small images.
  Lighting and occlusion in test images.
Overall, here were some of my observations
  Dropout helped avoid overfitting to easy classes and corrected some misclassifications.
  BatchNorm improved consistency but still misclassified underexposed or rotated samples.

L2 regularization smoothed training but had minimal impact on edge-case failures.


4. Comparison of Main & variant experiments
For the main experiment test accuracy is 74.2%, and it works pretty well/simple, though there is slight overfitting.
For the variation batch normalization test accuracy is 76.4% and it works faster and has more stable training, though it has more complexity.
For the variation L2 regularization the test accuracy is 75.6% and it slightly reduces overfitting, though there is minimal gain.
For dropout, test accuracy is 76.8%, making it the highest while it has the best generalization and avoids overfit though there is slow convergence.

Conclusion: In general, dropout offers the best improvement while BatchNorm stabilizes training and L2 has mild impact comparably.


