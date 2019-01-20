# MNIST-and-USPS-digit
Classification of MNIST and USPS digit images

In this project,I have MNIST dataset consisting of 60K images and used 10k images as a test set from MNIST dataset and 1500 images as a test set from USPS images. My models will be trained on 50k MNIST images and be evaluated on both MNIST test set(10k) and USPS test set(1.5k).

Here, I have implemented several types of classification models for classifying MNIST digit images and USPS images.

1.) Logistic Regression
2.) Multilayer Perceptron model
3.) Support vector Machine
4.) Random Forest

Along with it, I have used bagging and boosting to check whether accuracy gets increased or not.

Now the question comes why have I taken two test sets, one from MNIST and another one from USPS. The reason is to check whether my model supports “NO FREE LUNCH” theorem or not.


Conclusion:

During making this project, I got familiar with bagging and boosting concepts and what they actually do?
I came to know of bagging that it creates many models working together on a single dataset and aggregate its accuracy.
In boosting, It boosts the misclassified predictions that means we will give higher weights to weak learners until it falls in
correct classified region.
Along with it, I also knew about “NO FREE LUNCH” theorem which tells a great model for one problem may not hold for another problem, so it is better to try multiple models and find one that works best for a particular problem.
