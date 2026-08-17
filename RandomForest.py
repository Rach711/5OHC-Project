import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score, precision_score
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV


# Importing the datasets
axbend = pd.read_csv('axbend.csv', delimiter=" ")
axbend = axbend.rename({'V1': '3', 'V2': '4', 'V3': '5', 'V4': '6','V5': '7', 'V6': '8','V7': '9', 'V8': '10','V9': '11', 'V10': '12','V11': '13', 'V12': '14','V13': '15', 'V14': '16','V15': '17', 'V16': '18', 'V17': '19', 'V18': '20', 'V19': '21', 'V20': '22', 'V21': '23'}, axis='columns')
axbend = axbend.add_suffix('ad')
axbend = axbend.rename(columns={'V1ad':'ID'})
axbend

buckle = pd.read_csv('buckle.csv', delimiter=" ")
buckle = buckle.rename({'V1': '3', 'V2': '4', 'V3': '5', 'V4': '6','V5': '7', 'V6': '8','V7': '9', 'V8': '10','V9': '11', 'V10': '12','V11': '13', 'V12': '14','V13': '15', 'V14': '16','V15': '17', 'V16': '18', 'V17': '19', 'V18': '20', 'V19': '21', 'V20': '22', 'V21': '23'}, axis='columns')
buckle = buckle.add_suffix('be')


inclin = pd.read_csv('inclin.csv', delimiter=" ")
inclin = inclin.rename({'V1': '3', 'V2': '4', 'V3': '5', 'V4': '6','V5': '7', 'V6': '8','V7': '9', 'V8': '10','V9': '11', 'V10': '12','V11': '13', 'V12': '14','V13': '15', 'V14': '16','V15': '17', 'V16': '18', 'V17': '19', 'V18': '20', 'V19': '21', 'V20': '22', 'V21': '23'}, axis='columns')
inclin = inclin.add_suffix('in')
 

opening = pd.read_csv('opening.csv', delimiter=" ")
opening = opening.rename({'V1': '3', 'V2': '4', 'V3': '5', 'V4': '6','V5': '7', 'V6': '8','V7': '9', 'V8': '10','V9': '11', 'V10': '12','V11': '13', 'V12': '14','V13': '15', 'V14': '16','V15': '17', 'V16': '18', 'V17': '19', 'V18': '20', 'V19': '21', 'V20': '22', 'V21': '23'}, axis='columns')
opening = opening.add_suffix('og')


propel = pd.read_csv('propel.csv', delimiter=" ")
propel = propel.rename({'V1': '3', 'V2': '4', 'V3': '5', 'V4': '6','V5': '7', 'V6': '8','V7': '9', 'V8': '10','V9': '11', 'V10': '12','V11': '13', 'V12': '14','V13': '15', 'V14': '16','V15': '17', 'V16': '18', 'V17': '19', 'V18': '20', 'V19': '21', 'V20': '22', 'V21': '23'}, axis='columns')
propel = propel.add_suffix('pl')


rise = pd.read_csv('rise.csv', delimiter=" ")
rise = rise.rename({'V1': '3', 'V2': '4', 'V3': '5', 'V4': '6','V5': '7', 'V6': '8','V7': '9', 'V8': '10','V9': '11', 'V10': '12','V11': '13', 'V12': '14','V13': '15', 'V14': '16','V15': '17', 'V16': '18', 'V17': '19', 'V18': '20', 'V19': '21', 'V20': '22', 'V21': '23'}, axis='columns')
rise = rise.add_suffix('re')


roll = pd.read_csv('roll.csv', delimiter=" ")
roll = roll.rename({'V1': '3', 'V2': '4', 'V3': '5', 'V4': '6','V5': '7', 'V6': '8','V7': '9', 'V8': '10','V9': '11', 'V10': '12','V11': '13', 'V12': '14','V13': '15', 'V14': '16','V15': '17', 'V16': '18', 'V17': '19', 'V18': '20', 'V19': '21', 'V20': '22', 'V21': '23'}, axis='columns')
roll = roll.add_suffix('rl')


shear = pd.read_csv('shear.csv', delimiter=" ")
shear = shear.rename({'V1': '3', 'V2': '4', 'V3': '5', 'V4': '6','V5': '7', 'V6': '8','V7': '9', 'V8': '10','V9': '11', 'V10': '12','V11': '13', 'V12': '14','V13': '15', 'V14': '16','V15': '17', 'V16': '18', 'V17': '19', 'V18': '20', 'V19': '21', 'V20': '22', 'V21': '23'}, axis='columns')
shear = shear.add_suffix('sr')


shift = pd.read_csv('shift.csv', delimiter=" ")
shift = shift.rename({'V1': '3', 'V2': '4', 'V3': '5', 'V4': '6','V5': '7', 'V6': '8','V7': '9', 'V8': '10','V9': '11', 'V10': '12','V11': '13', 'V12': '14','V13': '15', 'V14': '16','V15': '17', 'V16': '18', 'V17': '19', 'V18': '20', 'V19': '21', 'V20': '22', 'V21': '23'}, axis='columns')
shift = shift.add_suffix('st')


slide = pd.read_csv('slide.csv', delimiter=" ")
slide = slide.rename({'V1': '3', 'V2': '4', 'V3': '5', 'V4': '6','V5': '7', 'V6': '8','V7': '9', 'V8': '10','V9': '11', 'V10': '12','V11': '13', 'V12': '14','V13': '15', 'V14': '16','V15': '17', 'V16': '18', 'V17': '19', 'V18': '20', 'V19': '21', 'V20': '22', 'V21': '23'}, axis='columns')
slide = slide.add_suffix('se')


stagger = pd.read_csv('stagger.csv', delimiter=" ")
stagger = stagger.rename({'V1': '3', 'V2': '4', 'V3': '5', 'V4': '6','V5': '7', 'V6': '8','V7': '9', 'V8': '10','V9': '11', 'V10': '12','V11': '13', 'V12': '14','V13': '15', 'V14': '16','V15': '17', 'V16': '18', 'V17': '19', 'V18': '20', 'V19': '21', 'V20': '22', 'V21': '23'}, axis='columns')
stagger = stagger.add_suffix('sg')


stretch = pd.read_csv('stretch.csv', delimiter=" ")
stretch = stretch.rename({'V1': '3', 'V2': '4', 'V3': '5', 'V4': '6','V5': '7', 'V6': '8','V7': '9', 'V8': '10','V9': '11', 'V10': '12','V11': '13', 'V12': '14','V13': '15', 'V14': '16','V15': '17', 'V16': '18', 'V17': '19', 'V18': '20', 'V19': '21', 'V20': '22', 'V21': '23'}, axis='columns')
stretch = stretch.add_suffix('sh')


tilt = pd.read_csv('tilt.csv', delimiter=" ")
tilt = tilt.rename({'V1': '3', 'V2': '4', 'V3': '5', 'V4': '6','V5': '7', 'V6': '8','V7': '9', 'V8': '10','V9': '11', 'V10': '12','V11': '13', 'V12': '14','V13': '15', 'V14': '16','V15': '17', 'V16': '18', 'V17': '19', 'V18': '20', 'V19': '21', 'V20': '22', 'V21': '23'}, axis='columns')
tilt = tilt.add_suffix('tt')


tip = pd.read_csv('tip.csv', delimiter=" ")
tip = tip.rename({'V1': '3', 'V2': '4', 'V3': '5', 'V4': '6','V5': '7', 'V6': '8','V7': '9', 'V8': '10','V9': '11', 'V10': '12','V11': '13', 'V12': '14','V13': '15', 'V14': '16','V15': '17', 'V16': '18', 'V17': '19', 'V18': '20', 'V19': '21', 'V20': '22', 'V21': '23'}, axis='columns')
tip = tip.add_suffix('tp')


twist = pd.read_csv('twist.csv', delimiter=" ")
twist = twist.rename({'V1': '3', 'V2': '4', 'V3': '5', 'V4': '6','V5': '7', 'V6': '8','V7': '9', 'V8': '10','V9': '11', 'V10': '12','V11': '13', 'V12': '14','V13': '15', 'V14': '16','V15': '17', 'V16': '18', 'V17': '19', 'V18': '20', 'V19': '21', 'V20': '22', 'V21': '23'}, axis='columns')
twist = twist.add_suffix('tw')


xdisp = pd.read_csv('xdisp.csv', delimiter=" ")
xdisp = xdisp.rename({'V1': '3', 'V2': '4', 'V3': '5', 'V4': '6','V5': '7', 'V6': '8','V7': '9', 'V8': '10','V9': '11', 'V10': '12','V11': '13', 'V12': '14','V13': '15', 'V14': '16','V15': '17', 'V16': '18', 'V17': '19', 'V18': '20', 'V19': '21', 'V20': '22', 'V21': '23'}, axis='columns')
xdisp = xdisp.add_suffix('xp')


ydisp = pd.read_csv('ydisp.csv', delimiter=" ")
ydisp = ydisp.rename({'V1': '3', 'V2': '4', 'V3': '5', 'V4': '6','V5': '7', 'V6': '8','V7': '9', 'V8': '10','V9': '11', 'V10': '12','V11': '13', 'V12': '14','V13': '15', 'V14': '16','V15': '17', 'V16': '18', 'V17': '19', 'V18': '20', 'V19': '21', 'V20': '22', 'V21': '23'}, axis='columns')
ydisp = ydisp.add_suffix('yp')


merged_df = pd.concat([axbend, buckle, inclin, opening, propel, rise, roll, shear, shift, slide, stagger, stretch, tilt, tip, twist, xdisp, ydisp], axis="columns")
pd.set_option('display.max_columns', None)
merged_df


column_names = list(merged_df.columns.values)
outcome_df = ['1', '1', '1', '1', '1', '1', '2', '2', '2', '2', '2', '2', '1', '1', '1', '2', '2', '2', '1', '1', '1', '1', '1', '1', '2', '2', '2', '1', '1', '1', '1', '1', '1', '2', '2', '2', '3', '3', '3', '3', '3', '3', '4', '4', '4', '4', '4', '4', '3', '3', '3', '4', '4', '4', '3', '3', '3', '3', '3', '3', '4', '4', '4', '3', '3', '3', '3', '3', '3', '4', '4', '4']  
total_data = merged_df.assign(Outcome=outcome_df)
total_data

#Establish X and y variables 
X=total_data.drop(['Outcome'], axis=1)
y=total_data['Outcome']


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=159625)


# Define hyperparamaters for Random Forest
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 1, stop = 100, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [2,4]
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the param grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(param_grid)

# Random Forest
rf_Model = RandomForestClassifier(bootstrap=False,
                                  max_depth=4,
                                  max_features='sqrt',
                                  min_samples_leaf=2,
                                  min_samples_split=2,
                                  n_estimators=100,
                                  random_state=159625)

rf_Model.fit(X_train,y_train)


# Predict the target variable for the test set
y_pred = rf_Model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
precision = precision_score(y_test, y_pred, average='weighted')
print("Precision:", precision)
f1 = f1_score(y_test, y_pred, average='weighted')
print("f1:", f1)
recall = recall_score(y_test, y_pred, average='weighted')
print("Recall:", recall)



# FEATURE SELECTION #1 - Lasso (L1)

# List of 20 predetermined random seeds
random_seeds = [685641, 249077, 18533, 426353, 622463, 103321, 396546, 427173, 286636, 335318, 785535, 231325, 405031, 390995, 37176, 755657, 101777, 517844, 969889, 159625]

# Iterate through the random seeds
for i, random_seed in enumerate(random_seeds, start=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    lasso = Lasso(alpha=1)

    # Fit the model on the training data
    lasso.fit(X_train, y_train)

    # Get the selected features and their coefficients
    selected_features = X_train.columns[lasso.coef_ != 0]
    feature_coefficients = lasso.coef_[lasso.coef_ != 0]

    # Create a DataFrame
    coeff_df = pd.DataFrame({'Feature': selected_features, 'Coefficient': feature_coefficients})

    # Save the DataFrame as a CSV file with a different name for each random seed
    filename = f'Lasso{i}.csv'
    coeff_df.to_csv(filename, index=False)

# FEATURE SELECTION #2 - Random Forest Importance Values

# Iterate through the random seeds
for i, random_seed in enumerate(random_seeds, start=1):
    np.random.seed(random_seed)  # Set the random seed


    # Create a Random Forest Classifier
    rf = RandomForestClassifier(bootstrap=False,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=2,
                                min_samples_split=2,
                                n_estimators=100,
                                random_state=random_seed)

    # Train the Random Forest model
    rf.fit(X_train, y_train)

    # Get feature importances
    feature_importances = rf.feature_importances_

    # Sort features by importance in descending order
    sorted_indices = feature_importances.argsort()[::-1]

    # Select the top features based on importance
    top_feature_indices = sorted_indices[:40]  # Selecting top 10 features, adjust as needed

    # Subset the original feature names using the selected indices
    selected_features = X_train.columns[top_feature_indices]
    selected_importances = feature_importances[top_feature_indices]

    # Create a DataFrame with selected features and importances
    coeff_df = pd.DataFrame({'Feature': selected_features, 'Importance': selected_importances})

    # Save the DataFrame as a CSV file with a different name for each random seed
    filename = f'RFCI{i}.csv'
    coeff_df.to_csv(filename, index=False)


# FEATURE SELECTION #3 - Recursive Feature Elimination with Cross-Validation

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np



# Iterate through the random seeds
for i, random_seed in enumerate(random_seeds, start=1):
    np.random.seed(random_seed)  # Set the random seed

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    # Create a Random Forest Classifier
    rf_Model = RandomForestClassifier(n_estimators=100,
                                      max_features='sqrt',
                                      max_depth=4,
                                      min_samples_split=2,
                                      min_samples_leaf=2,
                                      bootstrap=True,
                                      random_state=random_seed)

    # Perform Recursive Feature Elimination with Cross-Validation
    rfecv = RFECV(estimator=rf_Model, cv=5, step=1)  # Set the step value to 1 for forward selection
    X_train_selected = rfecv.fit_transform(X_train, y_train)

    # Get the selected features' indices
    selected_feature_indices = rfecv.get_support(indices=True)

    # Subset the original feature names using the selected indices
    selected_features = X_train.columns[selected_feature_indices]

    # Get feature importances from RFECV
    feature_importances = rfecv.estimator_.feature_importances_

    # Create a DataFrame with selected features and importances
    coeff_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importances})

    # Save the DataFrame as a CSV file with a different name for each random seed
    filename = f'RFECV_{i}.csv'
    coeff_df.to_csv(filename, index=False)


# FEATURE SELECTION #4 - Dispersion Ratio

# Iterate through the random seeds
for i, random_seed in enumerate(random_seeds, start=1):
    np.random.seed(random_seed)  # Set the random seed

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    am = np.mean(X_train, axis=0)
    gm = np.power(np.prod(X_train, axis=0), 1 / X_train.shape[0])
    disp_ratio = am / gm

    plt.bar(np.arange(X_train.shape[1]), disp_ratio, color='teal')

    # Save the plot as an image file with a different name for each random seed
    filename = f'DispersionRatio_{i}.png'
    plt.savefig(filename)
    plt.clf()  # Clear the plot for the next iteration


# FEATURE SELECTION #5 - Exhaustive Feature Selection with Cross-Validation (EFSCV) 
# Iterate through the random seeds
for i, random_seed in enumerate(random_seeds, start=1):
    np.random.seed(random_seed)  # Set the random seed

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create a Random Forest Classifier
    rf = RandomForestClassifier(bootstrap=False,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=2,
                                min_samples_split=2,
                                n_estimators=100,
                                random_state=random_seed)

    # Get all possible feature combinations
    all_features = X_train.columns
    num_features = len(all_features)
    selected_features = []

    for k in range(1, num_features + 1):
        for subset in combinations(all_features, k):
            X_train_selected = X_train[list(subset)]
            scores = cross_val_score(rf, X_train_selected, y_train, cv=5)  # Perform 5-fold cross-validation
            accuracy = np.mean(scores)

            if len(selected_features) == 0 or accuracy > selected_features[-1][0]:
                selected_features.append((accuracy, subset))

    # Sort the selected features by accuracy
    selected_features.sort(reverse=True)

    # Get the best feature subset
    best_accuracy, best_subset = selected_features[0]
    best_features = list(best_subset)

    # Create a DataFrame with selected features and accuracies
    coeff_df = pd.DataFrame({'Feature': best_features, 'Accuracy': best_accuracy})

    # Save the DataFrame as a CSV file with a different name for each random seed
    filename = f'EFSCV_{i}.csv'
    coeff_df.to_csv(filename, index=False)
