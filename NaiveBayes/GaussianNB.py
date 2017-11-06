import numpy as np
import pandas as pd
from collections import Counter


df = pd.read_csv('dataset/pima-indians-diabetes.csv')

df.dropna(inplace=True)

df1 = pd.DataFrame()
df1['num_times_pregnant'] = [8]
df1['plasma_glucose_concentration'] = [100]
df1['diastolic_blood_pressure'] = [85]
df1['triceps_skin_fold_thickness'] = [32]
df1['serum_insulin'] = [5]
df1['body_mass_index'] = [60]
df1['diabetes_pedigree_function'] = [0.225]
df1['Age'] = [58]


# print(df.corr())

num_classPositive = df['class'][df['class'] == 1].count()
num_classNegative = df['class'][df['class'] == 0].count()

total = len(df)


# Class: 1
Probb_testedPositive = num_classPositive/total
# print(Probb_testedPositive)

# Class: 0
Probb_testedNegative = num_classNegative/total
# print(Probb_testedNegative)


# Mean
data_mean = df.groupby('class').mean()
# print(data_mean)

# Variance
data_variance = df.groupby('class').var()
# print(data_variance)


# Means for Positive Tests
mean_num_times_pregnant = data_mean['num_times_pregnant'][data_variance.index == 1].values[0]
mean_plasma_glucose_concentration = data_mean['plasma_glucose_concentration'][data_variance.index == 1].values[0]
mean_diastolic_blood_pressure = data_mean['diastolic_blood_pressure'][data_variance.index == 1].values[0]
# mean_triceps_skin_fold_thickness = data_mean['triceps_skin_fold_thickness'][data_variance.index == 1].values[0]
# mean_serum_insulin = data_mean['serum_insulin'][data_variance.index == 1].values[0]
mean_body_mass_index = data_mean['body_mass_index'][data_variance.index == 1].values[0]
mean_diabetes_pedigree_function = data_mean['diabetes_pedigree_function'][data_variance.index == 1].values[0]
mean_Age = data_mean['Age'][data_variance.index == 1].values[0]


# Variance for Positive Tests
var_num_times_pregnant = data_variance['num_times_pregnant'][data_variance.index == 1].values[0]
var_plasma_glucose_concentration = data_variance['plasma_glucose_concentration'][data_variance.index == 1].values[0]
var_diastolic_blood_pressure = data_variance['diastolic_blood_pressure'][data_variance.index == 1].values[0]
# var_triceps_skin_fold_thickness = data_variance['triceps_skin_fold_thickness'][data_variance.index == 1].values[0]
# var_serum_insulin = data_variance['serum_insulin'][data_variance.index == 1].values[0]
var_body_mass_index = data_variance['body_mass_index'][data_variance.index == 1].values[0]
var_diabetes_pedigree_function = data_variance['diabetes_pedigree_function'][data_variance.index == 1].values[0]
var_Age = data_variance['Age'][data_variance.index == 1].values[0]



# Means for Negative Tests
num_times_pregnant_mean = data_mean['num_times_pregnant'][data_variance.index == 0].values[0]
plasma_glucose_concentration_mean = data_mean['plasma_glucose_concentration'][data_variance.index == 0].values[0]
diastolic_blood_pressure_mean = data_mean['diastolic_blood_pressure'][data_variance.index == 0].values[0]
# triceps_skin_fold_thickness_mean = data_mean['triceps_skin_fold_thickness'][data_variance.index == 0].values[0]
# serum_insulin_mean = data_mean['serum_insulin'][data_variance.index == 0].values[0]
body_mass_index_mean = data_mean['body_mass_index'][data_variance.index == 0].values[0]
diabetes_pedigree_function_mean = data_mean['diabetes_pedigree_function'][data_variance.index == 0].values[0]
Age_mean = data_mean['Age'][data_variance.index == 0].values[0]


# Variance for Negative Tests
num_times_pregnant_var = data_variance['num_times_pregnant'][data_variance.index == 0].values[0]
plasma_glucose_concentration_var = data_variance['plasma_glucose_concentration'][data_variance.index == 0].values[0]
diastolic_blood_pressure_var = data_variance['diastolic_blood_pressure'][data_variance.index == 0].values[0]
# triceps_skin_fold_thickness_var = data_variance['triceps_skin_fold_thickness'][data_variance.index == 0].values[0]
# serum_insulin_var = data_variance['serum_insulin'][data_variance.index == 0].values[0]
body_mass_index_var = data_variance['body_mass_index'][data_variance.index == 0].values[0]
diabetes_pedigree_function_var = data_variance['diabetes_pedigree_function'][data_variance.index == 0].values[0]
Age_var = data_variance['Age'][data_variance.index == 0].values[0]




# Create a function that calculates p(x | y):
def p_x_given_y(x, mean_y, variance_y):

    # Input the arguments into a probability density function
    p = 1/(np.sqrt(2*np.pi*variance_y)) * np.exp((-(x-mean_y)**2)/(2*variance_y))

    # return p
    return p



# Numerator of the posterior
# num1 = [Probb_testedPositive * p_x_given_y(df1['num_times_pregnant'][0], mean_num_times_pregnant, var_num_times_pregnant) * p_x_given_y(df1['plasma_glucose_concentration'][0], mean_plasma_glucose_concentration, var_plasma_glucose_concentration) * p_x_given_y(df1['diastolic_blood_pressure'][0], mean_diastolic_blood_pressure, var_diastolic_blood_pressure) * p_x_given_y(df1['triceps_skin_fold_thickness'][0], mean_triceps_skin_fold_thickness, var_triceps_skin_fold_thickness) * p_x_given_y(df1['serum_insulin'][0], mean_serum_insulin, var_serum_insulin) * p_x_given_y(df1['body_mass_index'][0], mean_body_mass_index, var_body_mass_index) * p_x_given_y(df1['diabetes_pedigree_function'][0], mean_diabetes_pedigree_function, var_diabetes_pedigree_function) * p_x_given_y(df1['Age'][0], mean_Age, var_Age)]
num1 = [Probb_testedPositive * p_x_given_y(df1['num_times_pregnant'][0], mean_num_times_pregnant, var_num_times_pregnant) * p_x_given_y(df1['plasma_glucose_concentration'][0], mean_plasma_glucose_concentration, var_plasma_glucose_concentration) * p_x_given_y(df1['diastolic_blood_pressure'][0], mean_diastolic_blood_pressure, var_diastolic_blood_pressure) * p_x_given_y(df1['body_mass_index'][0], mean_body_mass_index, var_body_mass_index) * p_x_given_y(df1['diabetes_pedigree_function'][0], mean_diabetes_pedigree_function, var_diabetes_pedigree_function) * p_x_given_y(df1['Age'][0], mean_Age, var_Age)]

print('num1: ',num1)

# Numerator
# num2 = Probb_testedNegative * p_x_given_y(df1['num_times_pregnant'][0], num_times_pregnant_mean, num_times_pregnant_var) * p_x_given_y(df1['plasma_glucose_concentration'][0], plasma_glucose_concentration_mean, plasma_glucose_concentration_var) * p_x_given_y(df1['diastolic_blood_pressure'][0], diastolic_blood_pressure_mean, diastolic_blood_pressure_var) * p_x_given_y(df1['triceps_skin_fold_thickness'][0], triceps_skin_fold_thickness_mean, triceps_skin_fold_thickness_var) * p_x_given_y(df1['serum_insulin'][0], serum_insulin_mean, serum_insulin_var) * p_x_given_y(df1['body_mass_index'][0], body_mass_index_mean, body_mass_index_var) * p_x_given_y(df1['diabetes_pedigree_function'][0], diabetes_pedigree_function_mean, diabetes_pedigree_function_var) * p_x_given_y(df1['Age'][0], Age_mean, Age_var)
num2 = Probb_testedNegative * p_x_given_y(df1['num_times_pregnant'][0], num_times_pregnant_mean, num_times_pregnant_var) * p_x_given_y(df1['plasma_glucose_concentration'][0], plasma_glucose_concentration_mean, plasma_glucose_concentration_var) * p_x_given_y(df1['diastolic_blood_pressure'][0], diastolic_blood_pressure_mean, diastolic_blood_pressure_var) * p_x_given_y(df1['body_mass_index'][0], body_mass_index_mean, body_mass_index_var) * p_x_given_y(df1['diabetes_pedigree_function'][0], diabetes_pedigree_function_mean, diabetes_pedigree_function_var) * p_x_given_y(df1['Age'][0], Age_mean, Age_var)

print('num2: ',num2)
