# Movie recommender
Collaborative Filtering is a recommendation algorithm that uses ratings or the behavior of users to make predictions.

This project uses an item based collaborative filtering approach to find similarities between items and predict user preferences. The dataset is obtained from MovieLens which contains ratings from 43k users on over 3.5k movies.

General Overview:

1.) Transform the dataset to a sparse matrix. The rows represent users and columns as movies.

2.) Create an item to item similarity matrix using cosine similarity. 

3.) Predict a user's ratings 
