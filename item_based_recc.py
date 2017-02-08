import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from time import time
from sklearn.metrics import mean_squared_error
from math import sqrt

class ItemItemRecommender(object):

    def __init__(self, neighborhood_size):
        self.neighborhood_size = neighborhood_size

    def fit(self, ratings_sparse):
        self.ratings_sparse = ratings_sparse
        self.n_users = ratings_sparse.shape[0]
        self.n_items = ratings_sparse.shape[1]
        self.item_sim_mat = cosine_similarity(self.ratings_sparse.T)
        self._set_neighborhoods()

    def _set_neighborhoods(self):
        '''
        Sorts the item similarity matrix from least to greatest similarity.

        Neighborhood contains the top similarities between items within the specified neighborhood size. 
        '''
        least_to_most_sim_indexes = np.argsort(self.item_sim_mat, 1)
        self.neighborhoods = least_to_most_sim_indexes[:, -self.neighborhood_size:]

    def pred_one_user(self, user_id, show_run_time=False):
        '''
        Generates rating predictions for a single user.

        Input: user_id

        Output: List of rating predictions for a single user.
        '''
        start_time = time()
        items_rated_by_this_user = self.ratings_sparse[user_id].nonzero()[1]
        # Define somewhere to put rating predictions
        predictions = np.zeros(self.n_items)
        for item_to_rate in range(self.n_items):
            relevant_items = np.intersect1d(self.neighborhoods[item_to_rate],
                                            items_rated_by_this_user,
                                            assume_unique=True)
            predictions[item_to_rate] = self.ratings_sparse[user_id, relevant_items] * \
                self.item_sim_mat[item_to_rate, relevant_items] / \
                self.item_sim_mat[item_to_rate, relevant_items].sum()
        if show_run_time:
            print("Execution time: %f seconds" % (time()-start_time))
        clean_predictions = np.nan_to_num(predictions) # Converts nans to 0
        return clean_predictions

    def pred_all_users(self, show_run_time=False):
        '''
        Generates rating predictions for all users.

        Output: An array containing rating predictions for every user.
        '''
        start_time = time()
        all_ratings = [
            self.pred_one_user(user_id) for user_id in range(self.n_users)]
        if show_run_time:
            print("Execution time: %f seconds" % (time()-start_time))
        return np.array(all_ratings)

    def top_n_recs(self, user_id, n):
        '''
        Predicts top n movie ratings for a single user.

        Input: user_id, n which specifies the number of recommendations  
        Output: List containing indicies representing top n recommendations
        '''
        pred_ratings = self.pred_one_user(user_id)
        item_index_sorted_by_pred_rating = list(np.argsort(pred_ratings))
        items_rated_by_this_user = self.ratings_sparse[user_id].nonzero()[1]
        unrated_items_by_pred_rating = [item for item in item_index_sorted_by_pred_rating
                                        if item not in items_rated_by_this_user]
        return unrated_items_by_pred_rating[-n:]

    def predict(self,test,all_preds):
        '''
        For each user, get's predictions on items not yet rated.

        Input: Test Data, Predictions for all n_users

        Output: RMSE Score
        '''
        preds = []
        for user_id in xrange(self.n_users):
            movies = test[test['user'] == user_id+1]['movie']
            for i,rating in enumerate(all_preds[user_id]):
                if i+1 in movies.values:
                    preds.append(rating)
        clean_preds = np.nan_to_num(preds) # convert nans to 0
        result = sqrt(mean_squared_error(test['rating'], clean_preds))
        return result


def get_ratings_data():
    '''
    Reads in train and test data and creates a sparse matrix from train data.

    Output: Train DataFrame, Test DataFrame, Sparse Train Matrix. 
    '''
    train = pd.read_table("data/ua.base", names=["user", "movie", "rating", "timestamp"])
    test = pd.read_table("data/ua.test", names =["user", "movie", "rating", "timestamp"])
    highest_user_id = train.user.max()
    highest_movie_id = train.movie.max()
    ratings_sparse_mat = sparse.lil_matrix((highest_user_id, highest_movie_id))
    for _, row in train.iterrows():
        ratings_sparse_mat[row.user-1, row.movie-1] = row.rating  # subtract 1 from id's to match 0 indexing
    return train, test, ratings_sparse_mat





if __name__ == "__main__":
    start_time = time()
    train_data, test_data, ratings_sparse = get_ratings_data()
    my_rec_engine = ItemItemRecommender(neighborhood_size=75)
    my_rec_engine.fit(ratings_sparse)
    all_predictions = my_rec_engine.pred_all_users()
    score = my_rec_engine.predict(test_data,all_predictions)
    print("Execution time: %f minutes" % ((time()-start_time)/60))
    print "RMSE: ", score
