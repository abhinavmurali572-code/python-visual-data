# Assignment: Visualization with Seaborn and Matplotlib - Titanic Dataset
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load Titanic dataset from seaborn
titanic = sns.load_dataset('titanic')
print(titanic.head())
print(titanic.info())

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10,6)

# 1. COUNTPLOT: Passenger Count by Embarked Port, separated by sex
plt.figure()
sns.countplot(data=titanic, x='embarked', hue='sex', palette='Set2')
plt.title('Passenger Count by Embarked Port - Separated by Sex', fontsize=14, fontweight='bold')
plt.xlabel('Port of Embarkation (S=Southampton, C=Cherbourg, Q=Queenstown)')
plt.ylabel('Count')
plt.legend(title='Sex')
plt.show()
# Observation: Most passengers boarded from Southampton (S), and majority were male. Female count is higher from Cherbourg compared to other ports.

# 2. SWARMPLOT: Age Distribution across Classes, categorized by sex
plt.figure()
sns.swarmplot(data=titanic, x='pclass', y='age', hue='sex', palette='coolwarm', dodge=True)
plt.title('Age Distribution across Passenger Classes by Sex', fontsize=14, fontweight='bold')
plt.xlabel('Passenger Class (1=First, 2=Second, 3=Third)')
plt.ylabel('Age')
plt.legend(title='Sex')
plt.show()
# Observation: First class has more older passengers and less children. Third class has wide age range with more younger people. Age distribution is similar for male and female but more male outliers.

# 3. PIE CHART: Gender Distribution using Matplotlib
gender_counts = titanic['sex'].value_counts()
plt.figure()
colors = ['#66b3ff','#ff9999']
explode = (0.1, 0)  # explode male slice to highlight
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', 
        explode=explode, colors=colors, shadow=True, startangle=90)
plt.title('Gender Distribution - Titanic Passengers', fontsize=14, fontweight='bold')
plt.axis('equal') # equal aspect ratio
plt.show()
# Observation: About 64.8% passengers were male and 35.2% female, showing male dominance on Titanic.

# 4. HEATMAP: Correlation Matrix
# Select numeric columns
corr_cols = ['age', 'fare', 'pclass', 'sibsp', 'parch', 'survived']
corr_matrix = titanic[corr_cols].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix - Age, Fare, Pclass, SibSp, Parch, Survived', fontsize=14, fontweight='bold')
plt.show()
# Observation: Fare and Pclass are strongly negatively correlated (-0.55). Survived is negatively correlated with Pclass (-0.34) meaning higher class had more survival chance. Fare positively correlated with survival (0.26).

# 5. VIOLIN PLOT: Fare by Class and Gender
plt.figure()
sns.violinplot(data=titanic, x='pclass', y='fare', hue='sex', split=True, palette='Set1')
plt.title('Fare Distribution by Class and Gender', fontsize=14, fontweight='bold')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.ylim(0, 300) # limit to see better
plt.legend(title='Sex')
plt.show()
# Observation: First class has highest fare range, with wide distribution. Fare is higher for females in first class median. Third class fare is very low and clustered. Some high outliers in all classes.

# 6. SCATTER PLOT: Age vs Fare colored by survival
plt.figure()
sns.scatterplot(data=titanic, x='age', y='fare', hue='survived', palette='viridis', alpha=0.7)
plt.title('Age vs Fare - Colored by Survival Status', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend(title='Survived (0=No, 1=Yes)')
plt.show()
# Observation: Younger passengers with high fare (first class) had higher survival. Many low-fare, young/middle-aged passengers did not survive. No clear linear relation between age and fare, but high fare outliers are mostly survivors.
