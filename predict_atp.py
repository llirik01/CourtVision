import pandas as pd
import warnings
from dateutil.relativedelta import relativedelta
from joblib import load
import xgboost as xgb

pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

# Load pre-trained machine learning models
logreg = load('./models_to_use/logreg.joblib')  # Logistic Regression model
rfc = load('./models_to_use/RFC.pkl')  # Random Forest Classifier

xgbc = xgb.XGBClassifier()
xgbc.load_model('./models_to_use/XGBoost.json')  # XGBoost model


def W(t, f=0.6):
    """
    Compute time discounting weight.

    :param t: int - Time difference in months.
    :param f: float - Discount factor (default is 0.6).
    :return: float - Discounted weight.
    """
    return round(min(f, f ** t), 10)


# List of statistical features used for predictions
stats_list = ['1st_in', '1st_ptwon', '2nd_ptwon', 'bp_saved', 'ace_per_s', 'df_per_s',
              '1st_r_ptwon', '2nd_r_ptwon', 'bp_won', 'tot_pt_won_s', 'tot_pt_won_r',
              'tot_pt_won', 'gms_won_s', 'gms_won_r', 'tot_gms_won']

# List of columns used for feature extraction and prediction
columns = ['p1_seed', 'p2_seed', 'ht_diff', 'rank_diff', 'rank_pts_diff',
           '1st_in_diff', '1st_ptwon_diff', '2nd_ptwon_diff', 'bp_saved_diff',
           'ace_per_s_diff', 'df_per_s_diff', '1st_r_ptwon_diff',
           '2nd_r_ptwon_diff', 'bp_won_diff', 'tot_pt_won_s_diff',
           'tot_pt_won_r_diff', 'tot_pt_won_diff', 'gms_won_s_diff',
           'gms_won_r_diff', 'tot_gms_won_diff']


def pred_proba(df, surface, p1_seed, p2_seed, p1_ht, p1_rank, p1_pts, p2_ht, p2_rank, p2_pts, date,
               p1_name, p2_name, statss_list=stats_list):
    """
    Predict match outcome probabilities.

    :param df: DataFrame - Historical match data.
    :param surface: str - Court surface type.
    :param p1_seed: str - Player 1 seed.
    :param p2_seed: str - Player 2 seed.
    :param p1_ht: int - Player 1 height.
    :param p1_rank: int - Player 1 ranking.
    :param p1_pts: int - Player 1 ranking points.
    :param p2_ht: int - Player 2 height.
    :param p2_rank: int - Player 2 ranking.
    :param p2_pts: int - Player 2 ranking points.
    :param date: datetime - Match date.
    :param p1_name: str - Player 1 name.
    :param p2_name: str - Player 2 name.
    :param statss_list: list - List of statistical features (default is stats_list).
    :return: tuple - Probabilities predicted by Logistic Regression, Random Forest, and XGBoost
                     for both all surfaces and a specific surface.
    """
    def get_info():
        """
        Extract basic player comparison features.

        :return: list - Player-specific features (seeds, height difference, ranking difference, points difference).
        """
        seed_mapping = {'LL': 0, 'Q': 1, 'S': 2, 'UN': 3, 'WC': 4}  # Mapping for seed values
        info_lst = [
            seed_mapping[p1_seed],
            seed_mapping[p2_seed],
            p1_ht - p2_ht,
            p1_rank - p2_rank,
            p1_pts - p2_pts
        ]

        return info_lst

    def predict_stats(dff, datee):
        """
        Predict player's statistics using weighted historical data.

        :param dff: DataFrame - Player's match data.
        :param datee: datetime - Reference date.
        :return: list - Predicted statistics for the player.
        """

        df_copy = dff.copy()

        # Calculate time difference in months for each match
        mnth = []
        for i in df_copy['tourney_date'].tolist():
            diff = relativedelta(datee, i)
            months = diff.years * 12 + diff.months
            mnth.append(round(months, 1))

        # Compute weights based on recency
        weights = []
        for i in mnth:
            w = W(i)
            if w == 0:
                w = 0.0000000001
            weights.append(w)

        s = sum(weights)

        # Normalize weights, sum must be 1
        weights_norm = [round(i / s, 10) for i in weights]

        df_copy['weights_norm'] = weights_norm

        # Apply weights to statistics
        for i in statss_list:
            df_copy[i] = df_copy[i] * df_copy['weights_norm']

        # Compute weighted statistics
        stats_pred = []
        for i in statss_list:
            stats_pred.append(round(df_copy[i].sum(), 2))

        return stats_pred

    def combined_predictors():
        """
        Generate feature vectors for both all surfaces and the selected surface.

        :return: tuple - Feature vectors for all surfaces and the selected surface.
        """
        part1 = get_info()

        # Filter historical data for both players (all surfaces & specific surface)
        df_p1_all = df[df['player_name'] == p1_name]
        df_p1_surf = df[(df['player_name'] == p1_name) & (df['surface'] == surface)]
        df_p2_all = df[df['player_name'] == p2_name]
        df_p2_surf = df[(df['player_name'] == p2_name) & (df['surface'] == surface)]

        # Compute statistics differences between players
        p1_stats_all = predict_stats(df_p1_all, datee=date)
        p2_stats_all = predict_stats(df_p2_all, datee=date)
        stats_diff_all = [round(el1 - el2, 2) for el1, el2 in zip(p1_stats_all, p2_stats_all)]

        p1_stats_surf = predict_stats(df_p1_surf, datee=date)
        p2_stats_surf = predict_stats(df_p2_surf, datee=date)
        stats_diff_surf = [round(el1 - el2, 2) for el1, el2 in zip(p1_stats_surf, p2_stats_surf)]

        fnl_vector_all = part1 + stats_diff_all
        fnl_vector_surf = part1 + stats_diff_surf

        return fnl_vector_all, fnl_vector_surf

    # Generate input feature vectors
    final_vector_all, final_vector_surf = combined_predictors()

    # Prepare DataFrames for model predictions
    df_vector_to_predict_all = pd.DataFrame(final_vector_all, index=columns).swapaxes('index', 'columns')
    df_vector_to_predict_surf = pd.DataFrame(final_vector_surf, index=columns).swapaxes('index', 'columns')

    # Make probability predictions using different models
    pred_lr_all = round(logreg.predict_proba(df_vector_to_predict_all)[:, 1][0], 2)
    pred_rfc_all = round(rfc.predict_proba(df_vector_to_predict_all)[:, 1][0], 2)
    pred_xgbc_all = round(xgbc.predict_proba(df_vector_to_predict_all)[:, 1][0], 2)

    pred_lr_surf = round(logreg.predict_proba(df_vector_to_predict_surf)[:, 1][0], 2)
    pred_rfc_surf = round(rfc.predict_proba(df_vector_to_predict_surf)[:, 1][0], 2)
    pred_xgbc_surf = round(xgbc.predict_proba(df_vector_to_predict_surf)[:, 1][0], 2)

    return pred_lr_all, pred_lr_surf, pred_rfc_all, pred_rfc_surf, pred_xgbc_all, pred_xgbc_surf


def winner_name(probability, name1, name2):
    """
     Determine the predicted winner based on probability.

     :param probability: float - Predicted probability of Player 1 winning.
     :param name1: str - Player 1 name.
     :param name2: str - Player 2 name.
     :return: tuple - Predicted winner name and formatted probability.
     """
    if probability >= 0.5:
        name = name1
        probability = format(probability, '.2f')
    else:
        name = name2
        probability = format(1 - probability, '.2f')

    return name, probability
