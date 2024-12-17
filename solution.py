import pandas as pd
import columnNames as c
from functions import fillMissingData, scaleRequiredColumns, ordinallyEncode

mentalHealthTraining = pd.read_csv('train.csv')

mentalHealthTraining = fillMissingData(mentalHealthTraining)

mentalHealthTraining = scaleRequiredColumns(mentalHealthTraining)

# Replace the garbage values with unknown or null values.
mentalHealthTraining[c.SLEEP_DURATION].replace('Pune', 'Unknown')
mentalHealthTraining[c.SLEEP_DURATION].replace('No', 'None')
mentalHealthTraining[c.SLEEP_DURATION].replace('Indore', 'Unknown')
mentalHealthTraining[c.SLEEP_DURATION].replace('Sleep_Duration', 'Unknown')
mentalHealthTraining[c.SLEEP_DURATION].replace('Work_Study_Hours', 'Unknown')

# Drop the columns that do not have any contribution to the predictions.
mentalHealthTraining = mentalHealthTraining.drop([c.NAME, c.ACADEMIC_PRESSURE, c.WORK_PRESSURE, c.STUDY_SATISFACTION, c.JOB_SATISFACTION, c.CGPA], axis=1)

from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
mentalHealthTraining = ordinallyEncode(encoder, mentalHealthTraining)

# The column depression is the result so we assign it to y and drop it from the x variable.
y = mentalHealthTraining[c.DEPRESSION]
x = mentalHealthTraining.drop([c.DEPRESSION], axis=1)

from xgboost import XGBClassifier
model = XGBClassifier(enable_categorical = True)
model.fit(x, y)

mentalHealthTesting = pd.read_csv('test.csv')

mentalHealthTesting.isna().sum()

mentalHealthTesting = fillMissingData(mentalHealthTesting)

mentalHealthTesting = scaleRequiredColumns(mentalHealthTesting)

mentalHealthTesting[c.SLEEP_DURATION].replace('Pune', 'Unknown')
mentalHealthTesting[c.SLEEP_DURATION].replace('No', 'None')
mentalHealthTesting[c.SLEEP_DURATION].replace('Indore', 'Unknown')
mentalHealthTesting[c.SLEEP_DURATION].replace('Sleep_Duration', 'Unknown')
mentalHealthTesting[c.SLEEP_DURATION].replace('Work_Study_Hours', 'Unknown')

mentalHealthTesting = mentalHealthTesting.drop([c.NAME, c.ACADEMIC_PRESSURE, c.WORK_PRESSURE, c.STUDY_SATISFACTION, c.JOB_SATISFACTION, c.CGPA], axis=1)

mentalHealthTesting = ordinallyEncode(encoder, mentalHealthTesting)
y_pred = model.predict(mentalHealthTesting)

# Write the result into result.csv.
result = []
for i in range(len(y_pred)):
    result.append([mentalHealthTesting['id'][i], y_pred[i]])
resultData = pd.DataFrame(result, columns = ['id', 'Depression'])
resultData.to_csv('result.csv', index=False)