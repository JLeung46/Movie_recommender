# Movie recommender
Collaborative Filtering is a recommendation algorithm that predicts user's preferences based on the ratings or the behavior of other users. The assumption here is that if two users have similar opinions on one item then the two user's are likely to have the same opinion on a different item. 

This project uses an item based collaborative filtering approach to find similarities between items to predict user preferences. The dataset is obtained from MovieLens which contains ratings from 43k users on over 3.5k movies.

General Project Overview:

1.) Transform the dataset to a sparse matrix. The rows represent users and columns as movies.

2.) Create an item to item similarity matrix using cosine similarity. 

3.) Sort the matrix from least to greatest similarity.

4.) Define a neighborhood size to represent the 'n' most similar items.
