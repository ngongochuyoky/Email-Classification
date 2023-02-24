# Email-Classification
## Introduction  
Classifying an email as spam or non-spam (Ham mail) the classification effect must be positive, and minimizing the error of classifying Ham mail into Spam mail.

- Spam classification is a two-class text classification problem, two text classes are Ham (valid mail) and Spam (spam).
- The initial sample document set is the messages that have been classified into spam and valid messages, the text to be classified is the received emails.
- We can rely on certain characteristics or attributes of an email to increase the efficiency of classification. The characteristics of an email such as: subject, sending address, message body, message with attachments or not... The more you take advantage of such information, the more accurate the classification ability, and the analysis results. type is also highly dependent on the size of the training sample set.

 Classification Algorithms:
 - Random Forest Classifier
 - Multinomial Naïve Bayes
 - Logistic Regression
 - K Nearest Neighbor
 - Decision Tree Classifier
 - Support Vector Machine (Linear)
 - Support Vector Machine (RBF)
 
## File Email Classification.ipynb
The models used are those available in the Sklearn library.

Program execution sequence:

- Natural language processing with datasets.
- Split 2 datasets, train and test.
- Train model of Sklearn library with training dataset.
- Evaluate model quality by graphs, accuracy,...
- Evaluate the models against each other, choose the best model for the spam email classification problem.

The program correctly classified spam
Evaluate 7 classification algorithms and choose the best algorithm for spam classification problem.

## File app.py
Build web apps with Streamlit

Web application with functionality:
- Predict spam emails by algorithm
- Evaluation of predictive models
![Danhgiachung](https://user-images.githubusercontent.com/54812014/174081737-dd5af780-8d25-4c26-a43f-b7cd4e12186d.PNG)
![Tungthuattoan](https://user-images.githubusercontent.com/54812014/174081744-bd04086f-c8b3-4bbe-b578-36e5c26f2325.PNG)
