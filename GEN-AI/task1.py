import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.sparse import hstack

# Load dataset
df = pd.read_csv("IMDb-India.csv", encoding="ISO-8859-1")

# Clean and select columns
df_model = df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Rating']].dropna(subset=['Rating'])
df_model[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']] = df_model[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']].fillna('unknown')
df_model['Actors'] = df_model['Actor 1'] + ', ' + df_model['Actor 2'] + ', ' + df_model['Actor 3']

# Use TF-IDF Vectorizer for better feature scaling
vectorizer_genre = TfidfVectorizer()
vectorizer_director = TfidfVectorizer()
vectorizer_actors = TfidfVectorizer()

X_genre = vectorizer_genre.fit_transform(df_model['Genre'])
X_director = vectorizer_director.fit_transform(df_model['Director'])
X_actors = vectorizer_actors.fit_transform(df_model['Actors'])

# Combine all features
X = hstack([X_genre, X_director, X_actors])
y = df_model['Rating'].astype(float)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
