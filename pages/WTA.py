import streamlit as st
import pandas as pd
import time
from joblib import load

# Load pre-trained machine learning models
logreg = load('./models_to_use/logreg_wta.joblib')
mlp = load('./models_to_use/mlp_wta.pkl')

st.set_page_config(
    page_title='WTA',
    layout='wide')

# Define feature columns used for predictions
columns = ['pts_diff', 'PS_odds_diff', 'Max_odds_diff', 'Avg_odds_diff', 'age_diff']


def get_prediction(p1_pts_, p2_pts_, p1_birthdate_, p2_birthdate_, p1_pin_coeff_, p2_pin_coeff_,
                   p1_max_coeff_, p2_max_coeff_, p1_avg_coeff_, p2_avg_coeff_, match_date_):
    """
    Predicts the probability of Player 1 winning using Logistic Regression and Multilayer Perceptron models.

    :param p1_pts_: int - Player 1 points
    :param p2_pts_: int - Player 2 points
    :param p1_birthdate_: datetime - Player 1 birthdate
    :param p2_birthdate_: datetime - Player 2 birthdate
    :param p1_pin_coeff_: float - Pinnacle odds for Player 1
    :param p2_pin_coeff_: float - Pinnacle odds for Player 2
    :param p1_max_coeff_: float - Maximum odds for Player 1
    :param p2_max_coeff_: float - Maximum odds for Player 2
    :param p1_avg_coeff_: float - Average odds for Player 1
    :param p2_avg_coeff_: float - Average odds for Player 2
    :param match_date_: datetime - Date of the match
    :return: tuple - Probabilities of Player 1 winning according to Logistic Regression and MLP models
    """
    def calculate_year_difference(birthdate, match_date_):
        """
        Calculate the difference in years between two dates.

        :param birthdate: datetime - Birthdate of the player
        :param match_date_: datetime - Date of the match
        :return: float - Age in years, rounded to one decimal place
        """
        delta_days = (match_date_ - birthdate).days
        full_years = delta_days // 365  # Повні роки
        remainder_days = delta_days % 365  # Залишок днів
        fraction_of_year = remainder_days / 365.25  # Частка від року

        return round(full_years + fraction_of_year, 1)  # Округлення до одного знаку після коми

    def get_vector():
        """
        Constructs the feature vector required for model predictions.

        :return: list - Feature vector containing points difference, odds differences, and age difference
        """
        p1_age = calculate_year_difference(birthdate=p1_birthdate_, match_date_=match_date_)
        p2_age = calculate_year_difference(birthdate=p2_birthdate_, match_date_=match_date_)
        age_diff = round(p1_age - p2_age, 1)

        pts_diff = int(p1_pts_ - p2_pts_)
        pin_diff = round(p1_pin_coeff_ - p2_pin_coeff_, 2)
        max_diff = round(p1_max_coeff_ - p2_max_coeff_, 2)
        avg_diff = round(p1_avg_coeff_ - p2_avg_coeff_, 2)

        return [pts_diff, pin_diff, max_diff, avg_diff, age_diff]

    vector_to_predict = get_vector()
    df_vector_to_predict = pd.DataFrame(vector_to_predict, index=columns).swapaxes('index', 'columns')

    pred_lr = round(logreg.predict_proba(df_vector_to_predict)[:, 1][0], 2)
    pred_mlp = round(mlp.predict_proba(df_vector_to_predict)[:, 1][0], 2)

    return pred_lr, pred_mlp


def winner_name(probability, name1, name2):
    """
    Determines the winner's name and probability based on the predicted probability.

    :param probability: float - Predicted probability of Player 1 winning
    :param name1: str - Name of Player 1
    :param name2: str - Name of Player 2
    :return: tuple - Winner's name and formatted probability
    """
    if probability >= 0.5:
        name = name1
        probability = format(probability, '.2f')
    else:
        name = name2
        probability = format(1 - probability, '.2f')

    return name, probability


# Sidebar for user input with match details
with st.sidebar:
    # form tourney
    formT = st.form("T_form")
    formT.markdown("<h2 style='text-align: center; color: black; font-weight: normal;'>Match info</h2>",
                   unsafe_allow_html=True)
    curr_date = formT.date_input('Match date')
    t_submit = formT.form_submit_button("Save :pushpin:", use_container_width=True)

    if t_submit:
        st.success("✅ Information has been successfully saved!")

    for _ in range(22):
        st.text(' ')
    st.divider()
    st.markdown("<h5 style='text-align: center; color: black; font-weight: normal;'>Created by Kyryl Shum 🧑‍💻</h5>",
                unsafe_allow_html=True)

c1, c2 = st.columns(2)  # Create two columns for player input forms

with c1:
    formP1 = st.form("P1_form")  # Player 1 input form
    formP1.markdown("<h5 style='text-align: center; color: black; font-weight: normal;'>Player 1 info</h5>",
                    unsafe_allow_html=True)
    p1_name = formP1.text_input("P1 name")
    p1_birthdate = formP1.date_input("Date of birth")
    p1_pts = formP1.number_input("Amount of points", value=0, step=1)
    p1_pin_coeff = formP1.number_input("Pinnacle coefficient", value=1., step=1., format='%.2f')
    p1_max_coeff = formP1.number_input("Maximum coefficient", value=1., step=1., format='%.2f')
    p1_avg_coeff = formP1.number_input("Average coefficient", value=1., step=1., format='%.2f')

    p1_submit = formP1.form_submit_button("Submit :tennis:", use_container_width=True)

    # Validate player 1 inputs
    if p1_submit and p1_pts >= 0 and p1_pin_coeff >= 1 and p1_max_coeff >= 1 and p1_avg_coeff >= 1:
        st.success("✅ The information for Player 1 has been successfully saved!")
    elif p1_pts < 0:
        st.error("❌ Please enter a valid number of points. Its value cannot be less than zero.")
    elif p1_pin_coeff < 1 or p1_max_coeff < 1 or p1_avg_coeff < 1:
        st.error("❌ Please enter a valid coefficient value. Its value cannot be less than one.")


with c2:
    formP2 = st.form("P2_form")  # Player 2 input form
    formP2.markdown("<h5 style='text-align: center; color: black; font-weight: normal;'>Player 2 info</h5>",
                    unsafe_allow_html=True)
    p2_name = formP2.text_input("P2 name")
    p2_birthdate = formP2.date_input("Date of birth")
    p2_pts = formP2.number_input("Amount of points", value=0, step=1)
    p2_pin_coeff = formP2.number_input("Pinnacle coefficient", value=1., step=1., format='%.2f')
    p2_max_coeff = formP2.number_input("Maximum coefficient", value=1., step=1., format='%.2f')
    p2_avg_coeff = formP2.number_input("Average coefficient", value=1., step=1., format='%.2f')

    p2_submit = formP2.form_submit_button("Submit :tennis:", use_container_width=True)

    # Validate player 2 inputs
    if p2_submit and p2_pts >= 0 and p2_pin_coeff >= 1 and p2_max_coeff >= 1 and p2_avg_coeff >= 1:
        st.success("✅ The information for Player 2 has been successfully saved!")
    elif p2_pts < 0:
        st.error("❌ Please enter a valid number of points. Its value cannot be less than zero.")
    elif p2_pin_coeff < 1 or p2_max_coeff < 1 or p2_avg_coeff < 1:
        st.error("❌ Please enter a valid coefficient value. Its value cannot be less than one.")

# Button to generate prediction
makePred = st.button('Make prediction :chart_with_upwards_trend:', use_container_width=True)

if makePred:
    with st.spinner('Predicting... Wait for it...'):
        time.sleep(3)

        # Predict match outcome using different models
        lr_proba, mlp_proba = get_prediction(p1_pts_=p1_pts, p2_pts_=p2_pts, p1_birthdate_=p1_birthdate,
                                             p2_birthdate_=p2_birthdate,
                                             p1_pin_coeff_=p1_pin_coeff, p2_pin_coeff_=p2_pin_coeff,
                                             p1_max_coeff_=p1_max_coeff,
                                             p2_max_coeff_=p2_max_coeff, p1_avg_coeff_=p1_avg_coeff,
                                             p2_avg_coeff_=p2_avg_coeff, match_date_=curr_date)

    # Display prediction results
    st.text('')
    st.markdown(
        "<h2 style='text-align: center; color: black; font-weight: normal;'>🥇 Winner Prediction Results 🥇</h2>",
        unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns(2)

    col1.markdown("<h4 style='text-align: center; color: black; font-weight: 200;'>Logistic Regression</h4>",
                  unsafe_allow_html=True)
    col1.divider()
    col2.markdown("<h4 style='text-align: center; color: black; font-weight: 100;'>Multilayer perceptron</h4>",
                  unsafe_allow_html=True)
    col2.divider()

    name_lr, proba_lr = winner_name(lr_proba, name1=p1_name, name2=p2_name)
    name_mlp, proba_mlp = winner_name(mlp_proba, name1=p1_name, name2=p2_name)

    col1.markdown(f"<h4 style='text-align: center; color: black; font-weight: 200; '>{name_lr}</h4>",
                  unsafe_allow_html=True)
    col1.markdown(f"<h4 style='text-align: center; color: black; font-weight: 200; '>Probability: {proba_lr}</h4>",
                  unsafe_allow_html=True)

    col2.markdown(f"<h4 style='text-align: center; color: black; font-weight: 200; '>{name_mlp}</h4>",
                  unsafe_allow_html=True)
    col2.markdown(f"<h4 style='text-align: center; color: black; font-weight: 200; '>Probability: {proba_mlp}</h4>",
                  unsafe_allow_html=True)

    st.balloons()
