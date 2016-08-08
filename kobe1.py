'''
This is the second model I built for the kaggle Kobe Bryant shot prediction.

Like the first, it uses scikit-learn's RandomForestRegressor to generate the
probabilities. I use all the same features as the first, however I parse
the dates of the game and match them with specific teams to create new
dummy variables. For Example, in his first game Portland2000 is True and cannot
be true for any season other than the 2000-2001 season.

The result is a sparse matrix.
'''

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as Woods
from sklearn.ensemble import RandomForestRegressor as DeepWoods

kobe = pd.read_csv('~data.csv', parse_dates = 'game_date')

# Pulls out the indicies for the shots that Kaggle took out. Turns it into a list
testShotsIndex = kobe['shot_made_flag'].index[kobe['shot_made_flag'].apply(
                    np.isnan)].tolist()

# since the shots don't start at 0, they start at 1, I need to up every index
# number up by 1
testShotsIndex = [testShotsIndex[i] + 1 for i in range(len(testShotsIndex))]

# Creating a new series in the dataframe that will be filled with nan values
kobe['away'] = np.nan

# creating a dummy variable that determines whether Kobe was away or at home
# maps that AtSign function to the matchup series and then assigns the True
# False values to that Away series  because I want to know if it was a home
# game or away game

kobe['away'] = map(lambda x: True if '@' in x else False, kobe['matchup'])

kobe['season'] = kobe['season'].map(lambda x: x[5:])
# what team did he face
kobe['other_team'] = kobe['opponent'] + kobe['season']

kobe['game_date'] = kobe['game_date'].str[:-3]

for month in kobe['game_date']:
    kobe[str(month)] = kobe['game_date'] == month

#---------------DROPPING UNWANTED VARIABLES-------------------------------------
dropVals = ['lat', 'lon', 'season', 'combined_shot_type', 'game_event_id',
            'game_id', 'seconds_remaining', 'shot_distance', 'shot_zone_area',
            'team_id', 'shot_zone_range', 'shot_type',
            'team_name', 'matchup', 'shot_id', 'opponent', 'game_date']

for value in dropVals:
    kobe = kobe.drop(value, 1)

for elem in kobe['action_type'].unique():
    kobe[str(elem)] = kobe['action_type'] == elem

for elem in kobe['other_team'].unique():
    kobe[str(elem)] = kobe['other_team'] == elem

dropVals = ['action_type', 'other_team', 'shot_zone_basic']

for value in dropVals:
    kobe = kobe.drop(value, 1)

# Creating the training set by dropping all the observations with missing
# values
kobeTrain = kobe.dropna()
kobeTrain = kobeTrain.reset_index() # resetting the index

# Creating an empty dataframe for the test data
kobeTest = pd.DataFrame()
# Pulling out the all the values of the original data set that aren't equal
# to a missed shot (Success and NA's) bc isnan doesn't work for some reason
kobeTest = kobe[kobe.shot_made_flag != 0]
# Removing the successes from the kobeTest dataframe
kobeTest = kobeTest[kobeTest.shot_made_flag != 1]
kobeTest = kobeTest.reset_index() # resetting those indices

assert list(kobeTrain.columns.values) == list(kobeTest.columns.values)

variableList = list(kobe.columns.values)
variableList.remove('shot_made_flag')


# What I want to train my model on
trainFeatures = kobeTrain[variableList].values

#----------------I removed opponent and date from this for the sake of time---------
#----------------I'll add it back in later--------------

#------Using the RandomForestRegressor which should return probabilities--------
blackForest = DeepWoods(n_estimators = 1000,
                        max_depth = 10, min_samples_split = 5)

madeShot = kobeTrain["shot_made_flag"].values

model5 = blackForest.fit(trainFeatures, madeShot)
model5.score(trainFeatures, madeShot)

testFeatures = kobeTest[variableList].values

# Making the predictions based on my model
model5Prediction = model5.predict(testFeatures)

# Turning the predictions into a dataframe with the probability of success
# and the index of the shot
Result5 = pd.DataFrame({'shot_id' : testShotsIndex,
                        'shot_made_flag' : model5Prediction})

Result5.shot_made_flag[Result5.shot_made_flag == 1] = .9999999999
Result5.shot_made_flag[Result5.shot_made_flag == 0] = .0000000001

# exporting the results to a .csv file so I can submit it to kaggle
Result5.to_csv('~/kobeResult.csv',index= False)
