import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import altair as alt
import plotly.graph_objects as go
import requests
from io import BytesIO

st.set_page_config(layout="wide",
                  page_title="xG Scout",
                  page_icon='⚽️')

Competition_name = 'Eliteserien'

# Sidebar filters
def sidebar_filters(df, data_type):
    seasons = sorted(df['Season'].unique())
    seasons.insert(0, 'All Seasons')
    st.sidebar.markdown(f"## {data_type} Filters")
    selected_season = st.sidebar.selectbox(f'Select a Season for {data_type}', seasons)
    return selected_season

# Filter data based on the selection
def filter_data(df, selected_season):
    if selected_season != 'All Seasons':
        df = df[df['Season'] == selected_season]
    return df



# File loading function
def load_data(file_url, season):
    r = requests.get(file_url)
    data = r.content

    df = pd.read_excel(BytesIO(data))
    df['Season'] = season
    return df

# Currency conversion function
def convert_currency_to_numeric(value):
    try:
        if isinstance(value, str):
            if 'm' in value:
                return float(value.replace('€', '').replace('m', '')) * 1000000
            elif 'k' in value:
                return float(value.replace('€', '').replace('k', '')) * 1000
    except ValueError:
        pass
    return 0

# Data processing function
def process_data(df):
    df['League'] = df['League'].fillna(df['Club Country'] + ' Non Pro')
    df['Transfer_Fee_Numeric'] = df['Transfer Fee'].apply(convert_currency_to_numeric)
    df['League'] = df['League'].apply(lambda x: 'Superligaen' if isinstance(x, str) and 'Superligaen' in x else x)
    df['League'] = df['League'].apply(lambda x: 'PostNord-ligaen' if isinstance(x, str) and 'PostNord-ligaen' in x else x)
    return df




# Load and process data
def load_and_process_data(file_paths, seasons):
    dfs = [process_data(load_data(fp, s)) for fp, s in zip(file_paths, seasons)]
    df_arrivals = pd.concat(dfs, ignore_index=True)
    return df_arrivals



# Display DataFrame in Streamlit
def display_data(df_arrivals):
    st.write(df_arrivals)

def Transfer_Fees_by_Season(df_arrivals):
    chart_data = df_arrivals.groupby('Season')['Transfer_Fee_Numeric'].sum().reset_index()

    fig = go.Figure(data=[
        go.Bar(name='Total Transfer Fees', x=chart_data['Season'], y=chart_data['Transfer_Fee_Numeric'])
    ])

    # Use the global variable Competition_name in the title
    fig.update_layout(
        title=f'Total Transfer Fees by Season for {Competition_name} Arrivals',
        xaxis_title='Season',
        yaxis_title='Total Transfer Fee',
        autosize=True,
        margin=dict(l=100),
    )

    st.plotly_chart(fig, use_container_width=True)

    
# Function to create and display the Plotly chart
def top_transfers_chart(df_arrivals):
    df_sorted = df_arrivals.sort_values(['Season', 'Transfer_Fee_Numeric'], ascending=[True, False])
    #Define amount of players
    top_transfers = df_sorted.groupby('Season').head(3)

    # Create a copy of the DataFrame to avoid the SettingWithCopyWarning
    top_transfers_copy = top_transfers.copy()

    # Create a new column 'Player and Team' 
    top_transfers_copy.loc[:, 'Player and Team'] = top_transfers_copy['Player Name'] + '<br>' + top_transfers_copy['Main Team']

    # Make player's name bold and increase its font size compared to the original club
    top_transfers_copy.loc[:, 'Player and Team'] = '<b>' + top_transfers_copy['Player Name'] + '</b><br><span style="font-size: 80%;">' + top_transfers_copy['Main Team'] + '</span>'

    # Get unique seasons sorted in ascending order
    seasons = sorted(top_transfers_copy['Season'].unique(), reverse=False)

    # Make a bar for each season
    bars = []
    for season in seasons:
        # Sort data in descending order of transfer fees
        data = top_transfers_copy[top_transfers_copy['Season'] == season].sort_values('Transfer_Fee_Numeric', ascending=True)

        bars.append(
            go.Bar(name=str(season), x=data['Transfer_Fee_Numeric'], y=data['Player and Team'], orientation='h')
        )

    # Make the figure
    fig = go.Figure(data=bars)

    # Update layout
    fig.update_layout(
        barmode='group',  # bars are grouped together
        title=f'Top 3 Transfers per Season for {Competition_name}',
        xaxis_title='Transfer Fee',
        yaxis_title='Player Name and Main Team',
        legend_title='Season',
        legend=dict(
            x=1,  # place legend outside of chart
            y=1,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)',
            traceorder='reversed'  # reverse the order of legend items
        ),
        autosize=True,  # keep autosize True
        height=1000,  # custom height
        margin=dict(l=100),  # increase left margin to fit long team names
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    


# Function to create and display the Plotly line chart
def nationality_change_chart(df_arrivals):
    # Get top 10 nationalities
    top_nationalities = df_arrivals[df_arrivals['Nationality'] != 'Norway']['Nationality'].value_counts().nlargest(10).index.tolist()

    # Filter DataFrame for top 10 nationalities
    df_arrivals = df_arrivals[df_arrivals['Nationality'].isin(top_nationalities)]

    # Group by season and nationality, count the number of players, and reset the index
    chart_data = df_arrivals.groupby(['Season', 'Nationality']).size().reset_index(name='Count')

    # Create a MultiIndex with all combinations of seasons and nationalities
    multi_index = pd.MultiIndex.from_product([chart_data['Season'].unique(), top_nationalities],
                                             names=['Season', 'Nationality'])

    # Reindex the DataFrame and fill missing counts with zero
    chart_data = chart_data.set_index(['Season', 'Nationality']).reindex(multi_index, fill_value=0).reset_index()

    # Pivot the DataFrame to have each nationality as a separate column
    pivot_data = chart_data.pivot(index='Season', columns='Nationality', values='Count').reset_index()

    # Create the plotly figure
    fig = go.Figure()

    # Add a line for each nationality
    for nationality in top_nationalities:
        fig.add_trace(go.Scatter(x=pivot_data['Season'], y=pivot_data[nationality],
                                 mode='lines', name=nationality))

    # Update layout
    fig.update_layout(
        title=f'Change of Top 10 Nationalities over the Seasons for {Competition_name}',
        xaxis_title='Season',
        xaxis=dict(tickmode='linear', dtick=1),  # Set tickmode to linear and dtick to 1 to have an incremental step of 1
        yaxis_title='Player Count',
        autosize=True
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
def display_players_by_nationality(df_arrivals):
    # Add selectbox for nationalities
    unique_nationalities = df_arrivals['Nationality'].unique()
    selected_nationality = st.selectbox('Select a Nationality', unique_nationalities, key='nationality_selectbox')

    # Add selectbox for seasons
    unique_seasons = sorted(df_arrivals['Season'].unique())
    unique_seasons.insert(0, 'All Seasons')
    selected_season = st.selectbox('Select a Season', unique_seasons, key='season_selectbox')

    # Filter data based on the selection
    if selected_season != 'All Seasons':
        players_of_selected_nationality = df_arrivals[(df_arrivals['Nationality'] == selected_nationality) & (df_arrivals['Season'] == selected_season)]
        st.subheader(f'Players of Nationality: {selected_nationality} in Season: {selected_season} | {Competition_name}')
    else:
        players_of_selected_nationality = df_arrivals[df_arrivals['Nationality'] == selected_nationality]
        st.subheader(f'Players of Nationality: {selected_nationality} in All Seasons | {Competition_name}')

    # Calculate the number of players and the average age
    num_players = len(players_of_selected_nationality)
    avg_age = players_of_selected_nationality['Age'].mean()

    # Display the number of players and the average age
    st.markdown(f'<p style="color:green;">Number of players: {num_players}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:green;">Average age: {avg_age:.2f}</p>', unsafe_allow_html=True)

    # Display DataFrame
    st.dataframe(players_of_selected_nationality)


def top_leagues_chart(df_arrivals):
    # Get top 10 leagues
    top_leagues = df_arrivals['League'].value_counts().nlargest(10).index.tolist()

    # Filter DataFrame for top 10 leagues
    df_arrivals = df_arrivals[df_arrivals['League'].isin(top_leagues)]

    # Group by season and league, count the number of players, and reset the index
    chart_data = df_arrivals.groupby(['Season', 'League']).size().reset_index(name='Count')

    # Create a MultiIndex with all combinations of seasons and leagues
    multi_index = pd.MultiIndex.from_product([chart_data['Season'].unique(), top_leagues],
                                             names=['Season', 'League'])

    # Reindex the DataFrame and fill missing counts with zero
    chart_data = chart_data.set_index(['Season', 'League']).reindex(multi_index, fill_value=0).reset_index()

    # Pivot the DataFrame to have each league as a separate column
    pivot_data = chart_data.pivot(index='Season', columns='League', values='Count').reset_index()

    # Create the plotly figure
    fig = go.Figure()

    # Add a line for each league
    for league in top_leagues:
        fig.add_trace(go.Scatter(x=pivot_data['Season'], y=pivot_data[league],
                                 mode='lines', name=league))

    # Update layout
    fig.update_layout(
        title=f'Change of Top 10 Leagues over the Seasons | {Competition_name}',
        xaxis_title='Season',
        xaxis=dict(tickmode='linear', dtick=1),  # Set tickmode to linear and dtick to 1 to have an incremental step of 1
        yaxis_title='Player Count',
        autosize=True
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def display_players_by_league(df_arrivals):
    # Add selectbox for leagues
    unique_leagues = df_arrivals['League'].unique()
    selected_league = st.selectbox('Select a League', unique_leagues, key='league_selectbox')

    # Add selectbox for seasons
    unique_seasons = sorted(df_arrivals['Season'].unique())
    unique_seasons.insert(0, 'All Seasons')
    selected_season = st.selectbox('Select a Season', unique_seasons, key='league_season_selectbox')

    # Filter data based on the selection
    if selected_season != 'All Seasons':
        players_of_selected_league = df_arrivals[(df_arrivals['League'] == selected_league) & (df_arrivals['Season'] == selected_season)]
        st.subheader(f'Players from League: {selected_league} in Season: {selected_season} | {Competition_name}')
    else:
        players_of_selected_league = df_arrivals[df_arrivals['League'] == selected_league]
        st.subheader(f'Players from League: {selected_league} in All Seasons | {Competition_name}')

    # Calculate the number of players and the average age
    num_players = len(players_of_selected_league)
    avg_age = players_of_selected_league['Age'].mean()

    # Display the number of players and the average age
    st.markdown(f'<p style="color:green;">Number of players: {num_players}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:green;">Average age: {avg_age:.2f}</p>', unsafe_allow_html=True)

    # Display DataFrame
    st.dataframe(players_of_selected_league)


###########################
#      Departures
###########################

# Function to load and process departure data
def load_and_process_departure_data(file_paths, seasons):
    dfs = [process_data(load_data(fp, s)) for fp, s in zip(file_paths, seasons)]
    df_departures = pd.concat(dfs, ignore_index=True)
    return df_departures

def display_data_departures(df_departures):
    st.write(df_departures)
    
def Transfer_Fees_by_Season_Departures(df_departures):
    chart_data = df_departures.groupby('Season')['Transfer_Fee_Numeric'].sum().reset_index()

    fig = go.Figure(data=[
        go.Bar(name='Total Transfer Fees', x=chart_data['Season'], y=chart_data['Transfer_Fee_Numeric'])
    ])

    fig.update_layout(
        title=f'Total Transfer Fees by Season for Departures | {Competition_name}',
        xaxis_title='Season',
        yaxis_title='Total Transfer Fee',
        autosize=True,
        margin=dict(l=100),
    )

    st.plotly_chart(fig, use_container_width=True)
    
# Function to create and display the Plotly chart
def top_departures_chart(df_departures):
    df_sorted = df_departures.sort_values(['Season', 'Transfer_Fee_Numeric'], ascending=[True, False])
    #Define amount of players
    top_transfers = df_sorted.groupby('Season').head(3)

    # Create a copy of the DataFrame to avoid the SettingWithCopyWarning
    top_transfers_copy = top_transfers.copy()

    # Create a new column 'Player and Team' 
    top_transfers_copy.loc[:, 'Player and Team'] = top_transfers_copy['Player Name'] + '<br>' + top_transfers_copy['Main Team']

    # Make player's name bold and increase its font size compared to the original club
    top_transfers_copy.loc[:, 'Player and Team'] = '<b>' + top_transfers_copy['Player Name'] + '</b><br><span style="font-size: 80%;">' + top_transfers_copy['Main Team'] + '</span>'

    # Get unique seasons sorted in ascending order
    seasons = sorted(top_transfers_copy['Season'].unique(), reverse=False)

    # Make a bar for each season
    bars = []
    for season in seasons:
        # Sort data in descending order of transfer fees
        data = top_transfers_copy[top_transfers_copy['Season'] == season].sort_values('Transfer_Fee_Numeric', ascending=True)

        bars.append(
            go.Bar(name=str(season), x=data['Transfer_Fee_Numeric'], y=data['Player and Team'], orientation='h')
        )

    # Make the figure
    fig = go.Figure(data=bars)

    # Update layout
    fig.update_layout(
        barmode='group',  # bars are grouped together
        title=f'Top 3 Departures per Season | {Competition_name}',
        xaxis_title='Transfer Fee',
        yaxis_title='Player Name and Main Team',
        legend_title='Season',
        legend=dict(
            x=1,  # place legend outside of chart
            y=1,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)',
            traceorder='reversed'  # reverse the order of legend items
        ),
        autosize=True,  # keep autosize True
        height=1000,  # custom height
        margin=dict(l=100),  # increase left margin to fit long team names
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


# Function to create and display the Plotly line chart for departures
def nationality_change_chart_departures(df_departures):
    # Get top 10 nationalities
    top_nationalities = df_departures[df_departures['Nationality'] != 'Norway']['Nationality'].value_counts().nlargest(10).index.tolist()

    # Filter DataFrame for top 10 nationalities
    df_departures = df_departures[df_departures['Nationality'].isin(top_nationalities)]

    # Group by season and nationality, count the number of players, and reset the index
    chart_data = df_departures.groupby(['Season', 'Nationality']).size().reset_index(name='Count')

    # Create a MultiIndex with all combinations of seasons and nationalities
    multi_index = pd.MultiIndex.from_product([chart_data['Season'].unique(), top_nationalities],
                                             names=['Season', 'Nationality'])

    # Reindex the DataFrame and fill missing counts with zero
    chart_data = chart_data.set_index(['Season', 'Nationality']).reindex(multi_index, fill_value=0).reset_index()

    # Pivot the DataFrame to have each nationality as a separate column
    pivot_data = chart_data.pivot(index='Season', columns='Nationality', values='Count').reset_index()

    # Create the plotly figure
    fig = go.Figure()

    # Add a line for each nationality
    for nationality in top_nationalities:
        fig.add_trace(go.Scatter(x=pivot_data['Season'], y=pivot_data[nationality],
                                 mode='lines', name=nationality))

    # Update layout
    fig.update_layout(
        title=f'Departures: Change of Top 10 Nationalities over the Seasons | {Competition_name}',
        xaxis_title='Season',
        xaxis=dict(tickmode='linear', dtick=1),  # Set tickmode to linear and dtick to 1 to have an incremental step of 1
        yaxis_title='Player Count',
        autosize=True
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    
# Function to display players by nationality for departures
def display_players_by_nationality_departures(df_departures):
    # Add selectbox for nationalities
    unique_nationalities = df_departures['Nationality'].unique()
    selected_nationality = st.selectbox('Select a Nationality', unique_nationalities, key='nationality_selectbox_departures')

    # Add selectbox for seasons
    unique_seasons = sorted(df_departures['Season'].unique())
    unique_seasons.insert(0, 'All Seasons')
    selected_season = st.selectbox('Select a Season', unique_seasons, key='season_selectbox_departures')

    # Filter data based on the selection
    if selected_season != 'All Seasons':
        players_of_selected_nationality = df_departures[(df_departures['Nationality'] == selected_nationality) & (df_departures['Season'] == selected_season)]
        st.subheader(f'Players of Nationality: {selected_nationality} in Season: {selected_season} (Departures) | {Competition_name}')
    else:
        players_of_selected_nationality = df_departures[df_departures['Nationality'] == selected_nationality]
        st.subheader(f'Players of Nationality: {selected_nationality} in All Seasons (Departures) | {Competition_name}')

    # Calculate the number of players and the average age
    num_players = len(players_of_selected_nationality)
    avg_age = players_of_selected_nationality['Age'].mean()

    # Display the number of players and the average age
    st.markdown(f'<p style="color:green;">Number of players: {num_players}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:green;">Average age: {avg_age:.2f}</p>', unsafe_allow_html=True)

    # Display DataFrame
    st.dataframe(players_of_selected_nationality)

    
# Function to create and display the Plotly line chart for departures
def top_leagues_chart_departures(df_departures):
    # Get top 10 leagues
    top_leagues = df_departures['League'].value_counts().nlargest(10).index.tolist()

    # Filter DataFrame for top 10 leagues
    df_departures = df_departures[df_departures['League'].isin(top_leagues)]

    # Group by season and league, count the number of players, and reset the index
    chart_data = df_departures.groupby(['Season', 'League']).size().reset_index(name='Count')

    # Create a MultiIndex with all combinations of seasons and leagues
    multi_index = pd.MultiIndex.from_product([chart_data['Season'].unique(), top_leagues],
                                             names=['Season', 'League'])

    # Reindex the DataFrame and fill missing counts with zero
    chart_data = chart_data.set_index(['Season', 'League']).reindex(multi_index, fill_value=0).reset_index()

    # Pivot the DataFrame to have each league as a separate column
    pivot_data = chart_data.pivot(index='Season', columns='League', values='Count').reset_index()

    # Create the plotly figure
    fig = go.Figure()

    # Add a line for each league
    for league in top_leagues:
        fig.add_trace(go.Scatter(x=pivot_data['Season'], y=pivot_data[league],
                                 mode='lines', name=league))

    # Update layout
    fig.update_layout(
        title='Change of Top 10 Leagues over the Seasons (Departures)',
        xaxis_title='Season',
        xaxis=dict(tickmode='linear', dtick=1),  # Set tickmode to linear and dtick to 1 to have an incremental step of 1
        yaxis_title='Player Count',
        autosize=True
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


# Function to display players by league for departures
def display_players_by_league_departures(df_departures):
    # Add selectbox for leagues
    unique_leagues = df_departures['League'].unique()
    selected_league = st.selectbox('Select a League', unique_leagues, key='league_selectbox_departures')

    # Add selectbox for seasons
    unique_seasons = sorted(df_departures['Season'].unique())
    unique_seasons.insert(0, 'All Seasons')
    selected_season = st.selectbox('Select a Season', unique_seasons, key='league_season_selectbox_departures')

    # Filter data based on the selection
    if selected_season != 'All Seasons':
        players_of_selected_league = df_departures[(df_departures['League'] == selected_league) & (df_departures['Season'] == selected_season)]
        st.subheader(f'Players to League: {selected_league} in Season: {selected_season} (Departures)')
    else:
        players_of_selected_league = df_departures[df_departures['League'] == selected_league]
        st.subheader(f'Players to League: {selected_league} in All Seasons (Departures)')

    # Calculate the number of players and the average age
    num_players = len(players_of_selected_league)
    avg_age = players_of_selected_league['Age'].mean()

    # Display the number of players and the average age
    st.markdown(f'<p style="color:green;">Number of players: {num_players}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:green;">Average age: {avg_age:.2f}</p>', unsafe_allow_html=True)

    # Display DataFrame
    st.dataframe(players_of_selected_league)


###########################
#      Display APP
###########################

def app():

    selection = st.sidebar.radio("Choose Analysis", ('Arrivals', 'Departures'), key='choose_analysis')

    file_paths = [
        'https://raw.githubusercontent.com/Mozi333/Elitesserien/main/data/eliteserien_arrivals_2016.xlsx',
        'https://raw.githubusercontent.com/Mozi333/Elitesserien/main/data/eliteserien_arrivals_2017.xlsx',
        'https://raw.githubusercontent.com/Mozi333/Elitesserien/main/data/eliteserien_arrivals_2018.xlsx',
        'https://raw.githubusercontent.com/Mozi333/Elitesserien/main/data/eliteserien_arrivals_2019.xlsx',
        'https://raw.githubusercontent.com/Mozi333/Elitesserien/main/data/eliteserien_arrivals_2020.xlsx',
        'https://raw.githubusercontent.com/Mozi333/Elitesserien/main/data/eliteserien_arrivals_2021.xlsx',
        'https://raw.githubusercontent.com/Mozi333/Elitesserien/main/data/eliteserien_arrivals_2022.xlsx'
    ]
    seasons = range(2016, 2023)

    if selection == 'Arrivals':
        st.header(f"{Competition_name} | Arrivals Analysis")
        df_arrivals = load_and_process_data(file_paths, seasons)
        selected_season_arrivals = sidebar_filters(df_arrivals, 'Arrivals')
        df_arrivals = filter_data(df_arrivals, selected_season_arrivals)
        Transfer_Fees_by_Season(df_arrivals)
        top_transfers_chart(df_arrivals)
        nationality_change_chart(df_arrivals)
        display_players_by_nationality(df_arrivals)
        top_leagues_chart(df_arrivals)
        display_players_by_league(df_arrivals)

    elif selection == 'Departures':
        st.header(f"{Competition_name} | Departures Analysis")
        departure_file_paths = [
            'https://raw.githubusercontent.com/Mozi333/Elitesserien/main/data/eliteserien_departures_2016.xlsx',
            'https://raw.githubusercontent.com/Mozi333/Elitesserien/main/data/eliteserien_departures_2017.xlsx',
            'https://raw.githubusercontent.com/Mozi333/Elitesserien/main/data/eliteserien_departures_2018.xlsx',
            'https://raw.githubusercontent.com/Mozi333/Elitesserien/main/data/eliteserien_departures_2019.xlsx',
            'https://raw.githubusercontent.com/Mozi333/Elitesserien/main/data/eliteserien_departures_2020.xlsx',
            'https://raw.githubusercontent.com/Mozi333/Elitesserien/main/data/eliteserien_departures_2021.xlsx',
            'https://raw.githubusercontent.com/Mozi333/Elitesserien/main/data/eliteserien_departures_2022.xlsx'
        ]
        df_departures = load_and_process_departure_data(departure_file_paths, seasons)
        selected_season_departures = sidebar_filters(df_departures, 'Departures')
        df_departures = filter_data(df_departures, selected_season_departures)
        Transfer_Fees_by_Season_Departures(df_departures)
        top_transfers_chart(df_departures)
        nationality_change_chart_departures(df_departures)
        display_players_by_nationality_departures(df_departures)
        top_leagues_chart_departures(df_departures)
        display_players_by_league_departures(df_departures)



if __name__ == "__main__":
    app()


