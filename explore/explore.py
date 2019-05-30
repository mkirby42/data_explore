#!/usr/bin/env python
"""
lambdata - a collection of Data Science helper functions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import classification_report, confusion_matrix

class explore:

    """A class for exploring dataframes."""

    def __init__(self, df):
        self.df = df

    def high_cardinality_check(n, df):
      """
      Given a cardinality limit (n) and a dataframe this function will search the
      dataframe for features above the cardinality limit, then create a dict
      from the results
      """

      feature_list = []

      cardinality_values = []

      for _ in range(len(df.columns)):

          column_cardinality = len(df[df.columns[_]].value_counts())
          feature_name = df.columns[_]

          if column_cardinality > n:
              feature_list.append(feature_name)
              cardinality_values.append(column_cardinality)


      high_cardinality_features = dict(zip(feature_list, cardinality_values))

      return high_cardinality_features


    def model_analysis(model, train_X, train_y, model_name):
      model_probabilities = model.predict_proba(train_X)

      Model_Prediction_Probability = []

      for _ in range(len(train_X)):
        x = max(model_probabilities[_])
        Model_Prediction_Probability.append(x)

      plt.figure(figsize=(15,10))
      sns.distplot(Model_Prediction_Probability)
      plt.title(model_name+'Prediction Probabilities')
      # Set x and y ticks
      plt.xticks(color='gray')
      #plt.xlim(.5,1)
      plt.yticks(color='gray')

      # Create axes object with plt. get current axes
      ax = plt.gca()

      # Set grid lines
      ax.grid(b=True, which='major', axis='y', color='black', alpha=.2)

      # Set facecolor
      ax.set_facecolor('white')

      # Remove box
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.spines['bottom'].set_visible(False)
      ax.spines['left'].set_visible(False)
      ax.tick_params(color='white')
      plt.show();

      model_predictions = model.predict(train_X)

      print('\n\n', classification_report(train_y, model_predictions,
                                target_names=['0-Poisonous', '1-Edible']))

      con_matrix = pd.DataFrame(confusion_matrix(train_y, model_predictions),
                                            columns=['Predicted Poison', 'Predicted Edible'],
                                            index=['Actual Poison', 'Actual Edible'])

      plt.figure(figsize=(15,10))
      sns.heatmap(data=con_matrix, cmap='cool');
      plt.title(model_name + 'Confusion Matrix')
      plt.show();

      #print('\n\n', con_matrix)
      return con_matrix
