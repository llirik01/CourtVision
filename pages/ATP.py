import streamlit as st
from datetime import timedelta
from predict_atp import *
import time

# Set the page configuration
st.set_page_config(
    page_title='ATP',
    layout='wide')

# Load ATP match data from CSV
data_atp = pd.read_csv('./atp_matches_database.csv')
data_atp['tourney_date'] = pd.to_datetime(data_atp['tourney_date'])

# Extract and sort unique player names
atp_names = data_atp[['player_name']]
atp_names.sort_values(by=['player_name'], inplace=True, ignore_index=True)
lst_names = list(atp_names['player_name'].unique())

with st.sidebar:
    # Tournament form
    formT = st.form("T_form")

    formT.markdown("<h2 style='text-align: center; color: black; font-weight: normal;'>Tourney info</h2>",
                   unsafe_allow_html=True)

    surface = formT.selectbox("Surface", options=['Hard', 'Clay', 'Grass', 'Carpet'])  # Select match surface type
    curr_date = formT.date_input('Match date')  # Select match date
    t_submit = formT.form_submit_button("Save :pushpin:", use_container_width=True)  # Submit tournament info

    if t_submit:
        st.success("✅ Information has been successfully saved!")

    st.info('The database with player statistics was last updated on April 15, 2024.', icon="ℹ️")

    for _ in range(11):
        st.text(' ')
    st.divider()
    st.markdown("<h5 style='text-align: center; color: black; font-weight: normal;'>Created by Kyryl Shum 🧑‍💻</h5>",
                unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="small")  # Create two columns for player inputs

with c1:
    p1_name = st.selectbox("P1 name", options=lst_names)  # Select Player 1

    formP1 = st.form("P1_form")  # Player 1 info form

    formP1.markdown("<h5 style='text-align: center; color: black; font-weight: normal;'>Player 1 info</h5>",
                    unsafe_allow_html=True)

    p1_ht = formP1.number_input("Height", value=0, step=1)
    p1_rank = formP1.number_input("Current rank", value=0, step=1)
    p1_pts = formP1.number_input("Current points", value=0, step=1)
    p1_seed = formP1.selectbox("Seed", options=['UN', 'S', 'Q', 'LL', 'WC'])
    p1_submit = formP1.form_submit_button("Submit :tennis:", use_container_width=True)  # Submit Player 1 data

    # Validation checks for Player 1
    if p1_submit and p1_ht >= 140 and p1_rank >= 0 and p1_pts >= 0:
        st.success("✅ The information for Player 1 has been successfully saved!")
    elif p1_submit and p1_ht < 140:
        st.error("❌ Please enter a valid player`s height. Its value cannot be less than 140.")
    elif p1_rank < 0:
        st.error("❌ Please enter a valid player`s rank. Its value cannot be less than zero.")
    elif p1_pts < 0:
        st.error("❌ Please enter a valid number of points. Its value cannot be less than zero.")


with c2:
    p2_name = st.selectbox("P2 Name", options=lst_names)  # Select Player 2

    formP2 = st.form("P2_form")  # Player 2 info form

    formP2.markdown("<h5 style='text-align: center; color: black; font-weight: normal;'>Player 2 info</h5>",
                    unsafe_allow_html=True)

    p2_ht = formP2.number_input("Height", value=0, step=1)
    p2_rank = formP2.number_input("Current rank", value=0, step=1)
    p2_pts = formP2.number_input("Current points", value=0, step=1)
    p2_seed = formP2.selectbox("Seed", options=['UN', 'S', 'Q', 'LL', 'WC'])
    p2_submit = formP2.form_submit_button("Submit :tennis:", use_container_width=True)  # Submit Player 2 data

    # Validation checks for Player 2
    if p2_submit and p2_ht >= 140 and p2_rank >= 0 and p2_pts >= 0:
        st.success("✅ The information for Player 2 has been successfully saved!")
    elif p2_submit and p2_ht < 140:
        st.error("❌ Please enter a valid player`s height. Its value cannot be less than 140.")
    elif p2_rank < 0:
        st.error("❌ Please enter a valid player`s rank. Its value cannot be less than zero.")
    elif p2_pts < 0:
        st.error("❌ Please enter a valid number of points. Its value cannot be less than zero.")

# Button to trigger prediction
makePred = st.button('Make prediction :chart_with_upwards_trend:', use_container_width=True)

if makePred:
    with st.spinner('Predicting... Wait for it...'):
        time.sleep(3)

    # Count the total number of matches played by each player
    p1_match_num_all = len(data_atp[data_atp['player_name'] == p1_name])
    p2_match_num_all = len(data_atp[data_atp['player_name'] == p2_name])

    # Count matches played by each player in the last 60 days
    p1_match_num_all_60 = len(data_atp[(data_atp['player_name'] == p1_name) & (
            data_atp['tourney_date'] >= pd.to_datetime(curr_date) - timedelta(days=60)) & (
                                               data_atp['tourney_date'] <= pd.to_datetime(curr_date))])
    p2_match_num_all_60 = len(data_atp[(data_atp['player_name'] == p2_name) & (
            data_atp['tourney_date'] >= pd.to_datetime(curr_date) - timedelta(days=60)) & (
                                               data_atp['tourney_date'] <= pd.to_datetime(curr_date))])

    # Count matches played on the selected surface by each player
    p1_match_num_surf = len(data_atp[(data_atp['player_name'] == p1_name) & (data_atp['surface'] == surface)])
    p2_match_num_surf = len(data_atp[(data_atp['player_name'] == p2_name) & (data_atp['surface'] == surface)])

    # Count matches played on the selected surface in the last 60 days
    p1_match_num_surf_60 = len(data_atp[(data_atp['player_name'] == p1_name) & (data_atp['surface'] == surface) & (
            data_atp['tourney_date'] >= pd.to_datetime(curr_date) - timedelta(days=60)) & (
                                                data_atp['tourney_date'] <= pd.to_datetime(curr_date))])
    p2_match_num_surf_60 = len(data_atp[(data_atp['player_name'] == p2_name) & (data_atp['surface'] == surface) & (
            data_atp['tourney_date'] >= pd.to_datetime(curr_date) - timedelta(days=60)) & (
                                                data_atp['tourney_date'] <= pd.to_datetime(curr_date))])

    # Predict match outcome using different models
    lr_all_prob, lr_surf_prob, rfc_all_prob, rfc_surf_prob, xgbc_all_prob, xgbc_surf_prob = pred_proba(data_atp,
                                                                                                       surface=surface,
                                                                                                       p1_seed=p1_seed,
                                                                                                       p2_seed=p2_seed,
                                                                                                       p1_ht=p1_ht,
                                                                                                       p1_rank=p1_rank,
                                                                                                       p1_pts=p1_pts,
                                                                                                       p2_ht=p2_ht,
                                                                                                       p2_rank=p2_rank,
                                                                                                       p2_pts=p2_pts,
                                                                                                       date=curr_date,
                                                                                                       p1_name=p1_name,
                                                                                                       p2_name=p2_name)

    st.text(' ')
    st.markdown(
        "<h2 style='text-align: center; color: black; font-weight: normal;'>🥇 Winner Prediction Results 🥇</h2>",
        unsafe_allow_html=True)
    st.divider()

    # Display match count statistics for both players
    col1, col2 = st.columns(2)

    col1.info(f' {p1_name} has played a total of {p1_match_num_all} matches on all surfaces, {p1_match_num_all_60} of \
              which were in the last 60 days.', icon="ℹ️")
    col1.info(f' {p2_name} has played a total of {p2_match_num_all} matches on all surfaces, {p2_match_num_all_60} of \
              which were in the last 60 days.', icon="ℹ️")

    col1.text('')
    col1.markdown("<h4 style='text-align: center; color: black; font-weight: 200;'>All surfaces</h4>",
                  unsafe_allow_html=True)

    col2.info(f' {p1_name} has played a total of {p1_match_num_surf} matches on a {surface} surface, \
              {p1_match_num_surf_60} of which were in the last 60 days.', icon="ℹ️")
    col2.info(f' {p2_name} has played a total of {p2_match_num_surf} matches on a {surface} surface, \
              {p2_match_num_surf_60} of which were in the last 60 days.', icon="ℹ️")

    col2.text('')
    col2.markdown(f"<h4 style='text-align: center; color: black; font-weight: 100; '>{surface} surface</h4>",
                  unsafe_allow_html=True)

    st.divider()

    # Display prediction results
    st.markdown("<h4 style='text-align: center; color: black; font-weight: 200;'>Logistic regression</h4>",
                unsafe_allow_html=True)

    name_lr_all, proba_lr_all = winner_name(lr_all_prob, name1=p1_name, name2=p2_name)
    name_lr_surf, proba_lr_surf = winner_name(lr_surf_prob, name1=p1_name, name2=p2_name)

    cl1, cl2 = st.columns(2)

    cl1.markdown(f"<h4 style='text-align: center; color: black; font-weight: 200; '>{name_lr_all}</h4>",
                 unsafe_allow_html=True)
    cl1.markdown(f"<h4 style='text-align: center; color: black; font-weight: 200; '>Probability: {proba_lr_all}</h4>",
                 unsafe_allow_html=True)

    cl2.markdown(f"<h4 style='text-align: center; color: black; font-weight: 200; '>{name_lr_surf}</h4>",
                 unsafe_allow_html=True)
    cl2.markdown(f"<h4 style='text-align: center; color: black; font-weight: 200; '>Probability: {proba_lr_surf}</h4>",
                 unsafe_allow_html=True)

    st.divider()
    st.markdown("<h4 style='text-align: center; color: black; font-weight: 200;'>Random Forest</h4>",
                unsafe_allow_html=True)

    name_rfc_all, proba_rfc_all = winner_name(rfc_all_prob, name1=p1_name, name2=p2_name)
    name_rfc_surf, proba_rfc_surf = winner_name(rfc_surf_prob, name1=p1_name, name2=p2_name)

    cll1, cll2 = st.columns(2)

    cll1.markdown(f"<h4 style='text-align: center; color: black; font-weight: 200; '>{name_rfc_all}</h4>",
                  unsafe_allow_html=True)
    cll1.markdown(f"<h4 style='text-align: center; color: black; font-weight: 200;'>Probability: {proba_rfc_all}</h4>",
                  unsafe_allow_html=True)

    cll2.markdown(f"<h4 style='text-align: center; color: black; font-weight: 200; '>{name_rfc_surf}</h4>",
                  unsafe_allow_html=True)
    cll2.markdown(f"<h4 style='text-align: center; color: black; font-weight: 200;'>Probability: {proba_rfc_surf}</h4>",
                  unsafe_allow_html=True)

    st.divider()
    st.markdown("<h4 style='text-align: center; color: black; font-weight: 200;'>XGBoost</h4>", unsafe_allow_html=True)

    name_xgbc_all, proba_xgbc_all = winner_name(xgbc_all_prob, name1=p1_name, name2=p2_name)
    name_xgbc_surf, proba_xgbc_surf = winner_name(xgbc_surf_prob, name1=p1_name, name2=p2_name)

    clll1, clll2 = st.columns(2)

    clll1.markdown(f"<h4 style='text-align: center; color: black; font-weight: 200; '>{name_xgbc_all}</h4>",
                   unsafe_allow_html=True)
    clll1.markdown(
        f"<h4 style='text-align: center; color: black; font-weight: 200; '>Probability: {proba_xgbc_all}</h4>",
        unsafe_allow_html=True)

    clll2.markdown(f"<h4 style='text-align: center; color: black; font-weight: 200; '>{name_xgbc_surf}</h4>",
                   unsafe_allow_html=True)
    clll2.markdown(
        f"<h4 style='text-align: center; color: black; font-weight: 200; '>Probability: {proba_xgbc_surf}</h4>",
        unsafe_allow_html=True)

    st.balloons()