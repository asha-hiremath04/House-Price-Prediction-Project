import pandas as pd  

# Load the dataset  
df = pd.read_csv(r'C:\\Users\\admin\\Desktop\\IBM Project\\kc_house_data.csv')  

# Print data types  
#print("Data Types:\n", df.dtypes)  


# Print missing values  
print("\nMissing Values:\n", df.isnull().sum())  

# Print duplicate rows count  
print("\nDuplicate Rows:", df.duplicated().sum())  

# Print summary statistics  
print("\nData Summary:\n", df.describe())  

df_numeric = df.drop(columns=['date'])
print(df_numeric.corr())

df['date'] = pd.to_datetime(df['date'])  # Convert to datetime format
df['year'] = df['date'].dt.year          # Extract year
df['month'] = df['date'].dt.month        # Extract month
df['day'] = df['date'].dt.day            # Extract day

# Now drop the original date column
df = df.drop(columns=['date'])

# Check correlation again
print(df.corr())

 #First question 
print(df.dtypes)


#Second question
df.drop(["id"], axis=1, inplace=True)  # Drop only "id"
print(df.describe())  # Get the statistical summary

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


#Question 3
import numpy as np

mean = df['bedrooms'].mean()
# df['bedrooms'].replace(np.nan, mean, inplace=True)
df['bedrooms'] = df['bedrooms'].replace(np.nan, mean)

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

floor_counts = df['floors'].value_counts().to_frame()
print(floor_counts)

#Question :4
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.boxplot(x="waterfront", y="price", data=df)
plt.xlabel("Waterfront View (0 = No, 1 = Yes)")
plt.ylabel("Price")
plt.title("Boxplot of House Prices by Waterfront View")
plt.savefig("waterfront_price_boxplot.png")  # Saves the plot as an image file
plt.show()


#Queston :5
import seaborn as sns
import matplotlib.pyplot as plt

# Create the regression plot
plt.figure(figsize=(8,6))
sns.regplot(x=df["sqft_above"], y=df["price"], scatter_kws={"alpha":0.5})

# Save the plot as an image
plt.savefig("sqft_above_price_regplot.png")  
plt.show()

correlation_value = df["sqft_above"].corr(df["price"])
print("Correlation between sqft_above and price:", correlation_value)


from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Ensure df is already defined and contains required columns
X = df[['sqft_living']]  # Independent variable
y = df['price']  # Dependent variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Linear Regression model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Predict on the test set
y_pred = lm.predict(X_test)

# Calculate R² score
r2 = r2_score(y_test, y_pred)
print("R-squared (R²) value:", r2)

# Plot regression line
plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color='blue', label="Actual Prices", alpha=0.5)
plt.plot(X_test, y_pred, color='red', label="Regression Line")
plt.xlabel("sqft_living")
plt.ylabel("Price")
plt.title("Linear Regression Model: sqft_living vs. Price")
plt.legend()
plt.savefig("linear_regression_sqft_living.png")  # Save the plot
plt.show()


#7th question
# Import necessary library
from sklearn.linear_model import LinearRegression

# Define the features and target variable
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view",
            "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]

X = df[features]  # Independent variables (features)
Y = df["price"]   # Dependent variable (target)

# Create and fit the model
lm = LinearRegression()
lm.fit(X, Y)

# Calculate R² score
r2_score = lm.score(X, Y)
print("R-squared (R²) value:", r2_score)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Define the features and target variable
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view",
            "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]

X = df[features]  # Independent variables
Y = df["price"]   # Dependent variable

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create and fit the model
lm = LinearRegression()
lm.fit(X_train, Y_train)

# Predict on test data
Y_pred = lm.predict(X_test)

# Calculate R² score on test set
r2 = r2_score(Y_test, Y_pred)
print("R-squared (R²) value:", r2)


#Question :8
# Import necessary libraries
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Define the features and target variable
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view",
            "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]

X = df[features]  # Independent variables (features)
Y = df["price"]   # Dependent variable (target)

# Create the pipeline
pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('polynomial', PolynomialFeatures(include_bias=False)),
    ('model', LinearRegression())
])
# Fit the pipeline
pipeline.fit(X, Y)

# Calculate R² score
r2_score = pipeline.score(X, Y)
print("R-squared (R²) value:", r2_score)



#Question :9
# Import necessary libraries
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Define the features and target variable
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view",
            "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]

X = df[features]  # Independent variables (features)
Y = df["price"]   # Dependent variable (target)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

# Create the Ridge regression model with alpha = 0.1
ridge_model = Ridge(alpha=0.1)

# Fit the model using training data
ridge_model.fit(x_train, y_train)

# Calculate R² score using test data
r2_score = ridge_model.score(x_test, y_test)
print("R-squared (R²) value on test data:", r2_score)

# question"10
# Import necessary libraries
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Define the features and target variable
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view",
            "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]

X = df[features]  # Independent variables (features)
Y = df["price"]   # Dependent variable (target)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

# Perform second-order polynomial transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# Create the Ridge regression model with alpha = 0.1
ridge_poly_model = Ridge(alpha=0.1)

# Fit the model using transformed training data
ridge_poly_model.fit(x_train_poly, y_train)

# Calculate R² score using transformed test data
r2_score_poly = ridge_poly_model.score(x_test_poly, y_test)
print("R-squared (R²) value on test data after polynomial transformation:", r2_score_poly)





