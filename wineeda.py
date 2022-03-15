
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
%matplotlib inline

#Now let's import the data file

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
wdf = pd.read_csv(url, sep=';')

#Let's explore the file and the data we have available in the file

wdf.sample(5)

wdf.info()

wdf.describe()

#Now let's check the data itself - do we have null values in any of the columns?

wdf.isnull().sum()

#Now let's check the cardinality of each column

wdf.nunique()

#We will group the wines into 3 categories - 'Poor', 'Good' and 'Excellent' based on their quality score, and that is in order to examine the data in a more user friendly way

wdf['quality group'] = wdf.quality.apply(lambda x:'Excellent' if x>=7 else 'Good' if (x>4 and x<7) else 'Poor')
wdf

plt.pie(wdf['quality group'].value_counts(), labels=wdf['quality group'].unique(), colors = sb.color_palette("pastel")[0:5], shadow = True, startangle = 100, radius = 1.3,textprops = {"fontsize":12}, autopct='%.0f%%')
my_circle=plt.Circle( (0,0), 1.1, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show();

#We can see that the majority of the wines is categorized as good with 82% of the wines, 14% are categorized as excellent and only 4% are categorized as poor

#We will present the distribution of the various parameters using histogram plot


wdf.hist(bins = 15, figsize=(15,15), color = "green")
plt.show();

#Next we will examine how the various parameters are correlated to the quality in order to check which parameter has effect on the quality of the wine

for i, col in enumerate(wdf.columns):
  if col != 'quality' and col != 'quality group':
    plt.figure(i, figsize=(10,5))
    sb.boxplot(x="quality group", y= col, order=['Poor','Good','Excellent'], data=wdf);
    ax = plt.gca();
    ax.set_title(col.upper());

#Next we will create a heatmap of the various parameters in the data to display the parameters correlation (negative/positive)

plt.figure(figsize=(17,7))
sb.heatmap(wdf.corr(), cmap="viridis", annot=True);

#Now let's explore each of the correlations we observed so far


sb.lmplot(x="citric acid", y="volatile acidity", hue="quality group", data=wdf);
ax = plt.gca();
ax.set_title("Citric Acid & Volatile Acidity");

sb.lmplot(x="fixed acidity", y="pH", hue="quality group", data=wdf);
ax = plt.gca();
ax.set_title("Fixed Acidity & pH");

sb.lmplot(x="citric acid", y="pH", hue="quality group", data=wdf);
ax = plt.gca();
ax.set_title("Citric Acid & pH");

sb.lmplot(x="density", y="alcohol", hue="quality group", data=wdf);
ax = plt.gca();
ax.set_title("Density & Alcohol");

sb.lmplot(x="density", y="fixed acidity", hue="quality group", data=wdf);
ax = plt.gca();
ax.set_title("Density & Fixed Acidity");

sb.lmplot(x="citric acid", y="fixed acidity", hue="quality group", data=wdf);
ax = plt.gca();
ax.set_title("Citric Acid & Fixed Acidity");

sb.lmplot(x="quality", y="alcohol", data=wdf);
ax = plt.gca();
ax.set_title("Quality & Alcohol");

sb.lmplot(x="total sulfur dioxide", y="free sulfur dioxide", hue="quality group", data=wdf);
ax = plt.gca();
ax.set_title("Total Sulfur Dioxide & Free Sulfur Dioxide");

sb.lmplot(x="sulphates", y="chlorides", hue="quality group", data=wdf);
ax = plt.gca();
ax.set_title("Sulphates & Chlorides");

sb.lmplot(x="fixed acidity", y="volatile acidity", hue="quality group", data=wdf);
ax = plt.gca();
ax.set_title("Fixed Acidity & Volatile Acidity");

sb.lmplot(x="volatile acidity", y="sulphates", hue="quality group", data=wdf);
ax = plt.gca();
ax.set_title("Volatile Acidity & Sulphates");

sb.lmplot(x="volatile acidity", y="quality", data=wdf);
ax = plt.gca();
ax.set_title("Volatile Acidity & Quality");

sb.lmplot(x="citric acid", y="density", hue="quality group", data=wdf);
ax = plt.gca();
ax.set_title("Citric Acid & Density");

sb.lmplot(x="citric acid", y="sulphates", hue="quality group", data=wdf);
ax = plt.gca();
ax.set_title("Citric Acid & Sulphates");

sb.lmplot(x="residual sugar", y="density", hue="quality group", data=wdf);
ax = plt.gca();
ax.set_title("Residual Sugar & Density");

sb.lmplot(x="chlorides", y="pH", hue="quality group", data=wdf);
ax = plt.gca();
ax.set_title("Chlorides & pH");

sb.lmplot(x="density", y="pH", hue="quality group", data=wdf);
ax = plt.gca();
ax.set_title("Density & pH");

sb.lmplot(x="sulphates", y="quality", data=wdf);
ax = plt.gca();
ax.set_title("Sulphates & Quality");

#T-Test

from scipy.stats.stats import ttest_ind

wdf['quality_category'] = wdf.quality.apply(lambda x:'exellent_quality' if x>=7 else'good_quality' if (x>4 and x<7) else 'poor_quality')
exellent_quality_group = wdf.quality_category == 'exellent_quality'
good_quality_group = wdf.quality_category == 'good_quality'
poor_quality_group = wdf.quality_category == 'poor_quality'




print(ttest_ind( wdf.loc[poor_quality_group].pH,wdf.loc[exellent_quality_group].pH, equal_var=False ,nan_policy = 'omit')) #pH
print(ttest_ind( wdf.loc[poor_quality_group].alcohol,wdf.loc[exellent_quality_group].alcohol, equal_var=False ,nan_policy = 'omit')) #alcohol
print(ttest_ind( wdf.loc[poor_quality_group].sulphates,wdf.loc[exellent_quality_group].sulphates, equal_var=False ,nan_policy = 'omit')) #sulphates
print(ttest_ind( wdf.loc[poor_quality_group].density,wdf.loc[exellent_quality_group].density, equal_var=False ,nan_policy = 'omit')) #density
print(ttest_ind( wdf.loc[poor_quality_group]['total sulfur dioxide'],wdf.loc[exellent_quality_group]['total sulfur dioxide'], equal_var=False ,nan_policy = 'omit')) #total_sulfur_dioxide
print(ttest_ind( wdf.loc[poor_quality_group]['free sulfur dioxide'],wdf.loc[exellent_quality_group]['free sulfur dioxide'], equal_var=False ,nan_policy = 'omit')) #free sulfur dioxide
print(ttest_ind( wdf.loc[poor_quality_group].chlorides,wdf.loc[exellent_quality_group].chlorides, equal_var=False ,nan_policy = 'omit')) #chlorides
print(ttest_ind( wdf.loc[poor_quality_group]['residual sugar'],wdf.loc[exellent_quality_group]['residual sugar'], equal_var=False ,nan_policy = 'omit')) #residual_sugar
print(ttest_ind( wdf.loc[poor_quality_group]['citric acid'],wdf.loc[exellent_quality_group]['citric acid'], equal_var=False ,nan_policy = 'omit')) #citric_acid
print(ttest_ind( wdf.loc[poor_quality_group]['volatile acidity'],wdf.loc[exellent_quality_group]['volatile acidity'], equal_var=False ,nan_policy = 'omit')) #volatile_acidity
print(ttest_ind( wdf.loc[poor_quality_group]['fixed acidity'],wdf.loc[exellent_quality_group]['fixed acidity'], equal_var=False ,nan_policy = 'omit')) #fixed_acidity