
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle


dataset = pd.read_excel("Book 1.xlsx")

# Split the data into features and target
features = dataset.drop("label", axis=1)

targets = dataset["label"]

# Split the data into a training and a testing set
train_features, test_features, train_targets, test_targets = \
        train_test_split(features, targets, train_size=0.75)# Train the model
tree = DecisionTreeClassifier(criterion="entropy",random_state=0)
tree = tree.fit(train_features, train_targets)
# Predict the classes of new, unseen data
prediction = tree.predict(test_features)
# print("Prediction: {}".format(prediction))

# Check the accuracy
score = tree.score(test_features, test_targets)
print("The prediction accuracy is: {:0.2f}%".format(score * 100))

with open ('savedModel.pkl','wb') as saveModel:
        pickle.dump( tree, saveModel)


