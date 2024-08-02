#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 22:13:52 2023

@author: mac
"""

# cd /Users/mac/Desktop/files/Data_Science_Python/spotify_nb
# streamlit run case_study_2.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import json
from scipy import stats
from ast import literal_eval
import streamlit as st
from streamlit_option_menu import option_menu
import itertools
st.set_page_config(layout="wide")

pd.options.plotting.backend = "plotly"

file_path = '/Users/mac/Desktop/files/Data_Science_Python/spotify_nb/'
spotify=pd.read_csv(file_path+'spotify_github.bz2', compression='bz2', low_memory=False)


spotify_graph=spotify.copy()

color_list=["#ff0000","#00ffff","#0000ff","#ccff33","#663300","#000000","#ff8080","#ff6600","#ffff00","#ff9999","#8080ff","#00ccff","#40ff00","#ffc61a","#000099","#994d00","#006600","#6600cc","#ff9900","#8533ff","#4700b3","#9933ff","#669900","#009900","#ff00ff","#ff99ff","#ff0066","#ffffff","#666699","#996633"]

level_order_dict = {k:['High','Decent','Low'] for k in spotify.columns if '_level' in k}

order_dict={"release_month":["January","February","March","April","May","June","July","August","September","October","November","December"],
            "art_pop_level":["Popular","Unpopular"]}
order_dict.update(level_order_dict)

num_cols = spotify.select_dtypes(include='number').columns
cat_cols = spotify.select_dtypes(include=['category','object']).columns


num_dict = {'acousticness':'Acousticness',
            'album_popularity':'Album Popularity',
            'artist_popularity':'Artist Popularity',
            'danceability':'Danceability',
            'duration_sec':'Duration (in seconds)',
            'energy':'Energy',
            'followers':'Number of Followers',
            'instrumentalness':'Instrumentalness',
            'liveness':'Liveness',
            'loudness':'Loudness',
            'release_year':'Released Year',
            'speechiness':'Speechiness',
            'tempo':'Tempo',
            'total_tracks':'Total Tracks',
            'track_number':'Track Number',
            'track_popularity':'Track Popularity',
            'valence':'Valence'
            }


cat_dict = {'acousticness_level':'Level of Acousticness',
            'album_type':'Album or Single',
            'artist_0':'Main Artist',
            'art_pop_level':'Artist Popularity Level',
            'danceability_level':'Level of Danceability',
            'energy_level':'Level of Energy',
            'explicit':'Explicit',
            'genres':'Genre',
            'key':'Key of the Song',
            'liveness_level':'Level of Liveness',
            'loudness_level':'Level of Loudness',
            'main_label':'Main Label',
            'mode':'Mode of the Song',
            'release_month':'Month of Release',
            'speechiness_level':'Level of Speechiness',
            'valence_level':'Level of Valence',
            'time_signature':'Time Signature'
            }


metrics_dict = {'acousticness':'Acousticness',
                'danceability':'Danceability',
                'energy':'Energy',
                'instrumentalness':'Instrumentalness',
                'liveness':'Liveness',
                'loudness':'Loudness',
                'speechiness':'Speechiness',
                'valence':'Valence'
                }

others_dict = {'album_popularity':'Album Popularity',
               'duration_sec':'Duration (in seconds)',
               'followers':'Number of Followers',
               'release_year':'Released Year',
               'tempo':'Tempo',
               'track_number':'Track Number',
               'track_popularity':'Track Popularity'
               }

all_dict = metrics_dict.copy()
all_dict.update(others_dict)


all_cols_dict = num_dict.copy()
all_cols_dict.update(cat_dict)
all_cols_dict.update({'track_name':'Song Title', 'artist_0':'Main Artist'})
all_cols_dict.update({'percent':'Percent', 'count':'Count'})


metrics = ['acousticness','danceability','energy','instrumentalness','liveness','loudness','speechiness','valence']
other_variables = np.setdiff1d(num_cols,metrics)

highly_correlated_metrics = np.setdiff1d(list(metrics_dict.values()), ['Acousticness','Instrumentalness','Loudness','Speechiness','Valence'])
highly_correlated_others = np.setdiff1d(list(others_dict.values()), ['Duration (in seconds)','Released Year','Tempo','Track Number'])
highly_correlated = list(highly_correlated_metrics) + list(highly_correlated_others)


melt_by_metrics = pd.melt(spotify.rename(columns=metrics_dict), id_vars=['art_pop_level'], value_vars=metrics_dict.values())
melt_by_highly_correlated_metrics = pd.melt(spotify.rename(columns=metrics_dict), id_vars=['art_pop_level'], value_vars=highly_correlated_metrics)

melt_by_others = pd.melt(spotify.rename(columns=others_dict), id_vars=['art_pop_level'], value_vars=others_dict.values())
melt_by_highly_correlated_others = pd.melt(spotify.rename(columns=others_dict), id_vars=['art_pop_level'], value_vars=highly_correlated_others)

melted = pd.melt(spotify.rename(columns=all_dict), id_vars=['art_pop_level'], value_vars=highly_correlated)



def fit_distribution(data):
    # estimate parameters
    mu = np.mean(data)
    sigma = np.std(data)
    # fit distribution
    dist = stats.norm(mu, sigma)
    return dist


def naive_bayes(df,predictors,tv='art_pop_level'):
    popular = df[df[tv]=='Popular'].copy()
    unpopular = df[df[tv]=='Unpopular'].copy()

    num_cols = np.setdiff1d(df.select_dtypes(include='number').columns, ["artist_popularity","duration_ms","released_day","rn","total_tracks"])
    cat_cols = np.setdiff1d(df.select_dtypes(include=['category','object']).columns,["album_id","album_name","artists","artist_genres","artist_id","artist_1","artist_2","artist_3","artist_4","artist_genres","genre_0","genre_1","genre_2","genre_3","genre_4","label","name","release_date","track_id","track_name","analysis_url","uri","type","track_href"])

    pop_prior = df[tv].value_counts(normalize=True).loc['Popular']
    unpop_prior = df[tv].value_counts(normalize=True).loc['Unpopular']
    
    pop_dict = {}
    un_dict = {}

    for c in predictors:
        if c in num_cols:
            pop_dict.update({c:fit_distribution(popular[c]).pdf(df[c])})
            un_dict.update({c:fit_distribution(unpopular[c]).pdf(df[c])})
        elif c in cat_cols:
            p_con = dict(popular[c].value_counts(normalize=True))
            un_con=dict(unpopular[c].value_counts(normalize=True))
            pop_dict[c] = df[c].map(p_con)
            un_dict[c] = df[c].map(un_con)
    
    pop_prob = pd.DataFrame(pop_dict).prod(axis=1) * pop_prior
    unpop_prob = pd.DataFrame(un_dict).prod(axis=1) * unpop_prior

    predicted = (pop_prob > unpop_prob).map({True:'Popular',False:'Unpopular'})
    results = pd.concat([predicted.reset_index(drop=True),df[tv].reset_index(drop=True)],axis=1)
    
    results.columns = ['Predicted','Actual']

    accuracy = np.mean(results['Predicted'] == results['Actual'])
    
    return results, accuracy


spotify_chi2 = pd.read_csv(file_path+'spotify_chi2.csv', index_col=0)
spotify_chi2.index.name = None






with st.sidebar: 
	selected = option_menu(
		menu_title = 'Navigation Pane',
		options = ['Abstract', 'Background Information', 'Data Cleaning','Exploratory Analysis', 'Testing Naive Bayes Predictions', 'Data Analysis', 'Conclusion', 'Bibliography'],
		menu_icon = 'music-note-list',
		icons = ['bookmark-check', 'book', 'box', 'map', 'boxes', 'bar-chart', 'check2-circle', 'book'],
		default_index = 0,
		)


if selected=='Abstract':
    st.title("Spotify Abstract")
    
    st.markdown("The dataset we will be using consists of information about thousands of tracks on Spotify. It contains several variables, including tracks by popular and unpopular artists, the artists of each track, the genres of each track, the labels under which the artists released the tracks, and musical attributes like danceability, valence, and energy. The target variable is the last variable, 'art_pop_level,' representing the popularity of the track’s artist, with the values “popular” and “unpopular”.")
    st.markdown("In this case study, we will analyze data of thousands of Spotify tracks available as of 2023. We will examine the patterns and correlations between each variable and the target variable 'art_pop_level' to determine which factors most likely influence the popularity of the artist. We will create a prediction model using the Naive Bayes algorithm to predict whether the artist is popular or unpopular based on selected variables (predictors) in the dataset.")
    
    st.markdown('<p style="font-size:14px"><i>Note: This app is best viewed in light mode, click on settings and then select "light"</i></p>',unsafe_allow_html=True)





if selected=="Background Information":
    st.title("Background Information")
    
    st.markdown("Spotify is currently one of the biggest music streaming platforms. As of 2023, Spotify has over 100 million tracks and more than 574 million users, including 226 million subscribers.<sup>1</sup> The dataset we’ll be examining contains tracks within the top 25 popular music genres (including pop, rock, hip hop, etc.).",unsafe_allow_html=True)
    
    st.markdown("Spotify for Artists, the platform that artists use, displays basic information about an artist’s track performance, including the number of streams, listeners, followers, top songs, etc. However, it does not account for key factors such as track attributes (danceability, valence, energy, etc.) and the popularity index of the tracks and artists themselves. Meanwhile, the Spotify algorithm uses these hidden indicators to recommend songs.<sup>2</sup>",unsafe_allow_html=True)
    st.markdown("SSpotify’s Popularity Index ranks an artist’s popularity relative to other artists on a scale from 0 to 100. While the Popularity Index is mainly determined by the recent number of streams, other factors (saves, shares, likes, playlists) can also influence it.<sup>3</sup> Since Spotify generates “popular tracks” based on the number of all-time and recent streams<sup>4</sup>, we can assume that track popularity can also influence an artist’s popularity.",unsafe_allow_html=True)
    st.markdown("According to previous studies by experts, the most common genre on Spotify (as of 2023) is pop, followed by hip-hop, but track popularity isn’t necessarily related to its musical genre, as there can be popular tracks across all genres. Genres like pop, hip-hop, and R&B can increase an artist’s chance of higher popularity and chart performance.<sup>5</sup> As of 2023, the top genres on Spotify are pop, EDM, rap, and rock<sup>6</sup>.",unsafe_allow_html=True)
    st.markdown("Though there are statements about popular artists today having high levels of danceability, energy, and valence, these do not guarantee artist popularity and success. While no definitive relationship is established between popularity and valence, more popular artists tend to have more energetic and danceable tracks. <sup>7</sup>", unsafe_allow_html=True)
    st.markdown("It is also said that artists under major labels can significantly outperform artists with indie labels in terms of overall popularity.<sup>8</sup> Due to the greater purchasing power of major labels, they can afford big advances, fancy packaging, and large promotional budgets, boosting an artist’s popularity upon an album or track’s release.<sup>9</sup> This case study also aims to test these claims to examine the factors most likely affecting artist popularity.",unsafe_allow_html=True)    




if selected=="Data Cleaning":
    st.title('Data Cleaning')
    st.markdown("The data cleaning process mainly involves cleaning music genres (organizing smaller subgenres into larger genre categories), cleaning artist labels (converting sublabels into major labels of which they are from), and adding a variable to measure the level of each musical attribute.")
    
    st.header('Genre Cleaning')
    
    st.markdown("As there are far too many subgenres in the dataset (pop rap, pop dance, west cost rap, etc.), I had to convert them into their main genres (such as pop, rock, hip hop, edm). Initially, all the music subgenres per track were stored in a list format, so I had to use the library 'literal_eval' and the 'pd.DataFrame.explode()' function to take all the subgenres out of the list format to organize and clean them.")
    st.markdown("By cleaning music genres, I have matched the specific subgenres in the dataset with their main genres using a dictionary 'genre_dict', with each key being the main genre and each value being the list of subgenres. Using the dictionary, I have replaced each of the subgenres that appear in the dataset as its corresponding main genre. This allows for broader categories when exploring Spotify tracks by music genres in the next section.")
    with st.expander("Click to view the genre cleaning code"):
        st.code('''# Use literal_eval to take all the genres out of the list format and stack them as a separate column
artist_genres_explode = spotify['artist_genres'].apply(literal_eval).explode()
# Then, rename the column as 'genres'
artist_genres_explode.rename('genres',inplace=True)
# Merge the spotify dataframe with the new genres column
spotify_merged = pd.merge(spotify,artist_genres_explode,left_index=True,right_index=True)

# Make the dictionary for all subgenres in the dataset to match their main genres
genre_dict = {'blues':['blues','blues rock','chicago blues','country blues','delta blues','desert blues','electric blues','jazz blues','modern blues','punk bluess','rhythm and blues','r&b','southern rock','texas blues'],
              'classical music':['renaissance','baroque','british contemporary classical','british modern classical','german baroque','classical','classical era','early modern classical','classical performance','early romantic era','english baroque','german romanticism','italian romanticism','late romantic era','impressionism','romantic','post-romantic era','contemporary classical','new romantic','neoclassicism','orchestral performance','violin','russian romanticism','opera','operetta','classical soprano','italian soprano','classical tenor','italian tenor','classical baritone'],
              'christian':['christian music','ccm','gospel','orthodox chant'],
              'choir':['british choir','cambridge choir','cathedral choir','choral',"children's choir",'south african choral','university choir'],
              'country':['alternative country','australian country','bluegrass','classic country pop','classic texas country','classic oklahoma country','contemporary country','country dawn','country pop','country rap','country road','country rock','modern country rock','nashville sound','outlaw country','progressive country','southern rock','texas country','western swing'],
              'cover':['classify','bardcore','fake','piano cover'],
              'dance':['arkansas country','edm','acid jazz','alternative dance','ballet','ballroom dance','boleros','bolero','cha-cha-cha','dance pop','dance punk','dance rock','dansband','disco','filmi','lounge','mambo','polka','pop dance','post-disco','uk dance','rumba','salsa','tango','twist','waltz'],
              'disco':['eurodisco','italo disco','nu disco','post-disco'],
              'early music':['early music','early music ensemble'],
              'edm':['breakbeat','drum and bass','dubstep','brostep','dance pop','dutch edm','electroclash','electronic trap','edm','freestyle','futurepop','grime','hardcore','hardstyle','electro house','house','big room','slap house','progressive house','jungle','moombahton','new beat','nu-disco','techno','trance'],
              'electronic':['ambient','compositional ambient','edm','dance pop','downtempo','dub','electronic rock','electro house','electropop','electropunk','folktronica','hyperpop','indietronica','industrial','metropopolis','new wave','noise','synth-pop','trip hop'],
              'experimental':['avant-garde','avant-pop','free improvisation','industrial','math rock','mellow gold','noise','neofolk','pov: indie','experimental pop','experimental rock'],
              'flamenco':['flamenco'],
              'folk':['ethnic','ectofolk','maritime','mariachi','music hall','norteno','folk-pop','folk pop','folk','folk rock','skiffle','sierreno','traditional','ranchera'],
              'funk':['acid jazz','funk','funk rock','g funk','go-go','jazz funk','p funk','synth funk','minneapolis sound'],
              'gospel':['gospel','gospel r&b','gospel soul','gospel rap'],
              'gothic':['cold wave','gothic country','dark wave','death rock','ethereal wave','gothabily','gothic metal','gothic rock'],
              'heavy metal':['alternative metal','avant-garde metal','birmingham metal','black metal','christian metal','death metal','djent','doom metal','extreme metal','folk metal','funk metal','glam metal','gothic metal','groove metal','heavy metal','industrial metal','melodic death metal','metal','metalcore','nu metal','proto-metal','power metal','progressive metal','rap metal','sludge metal','speed metal','stoner rock','symphonic metal','thrash metal','viking metal'],
              'hip hop':['rap','alternative hip hop','atl hip hop','atl trap','cali rap','canadian hip hop','chicago rap','comedy hip hop','conscious hip hop','country rap','crunk','detroit hip hop','dirty rap','dirty south rap','drill','east coast hip hop','electro','emo rap','experimental hip hop','gangster rap','g funk','gangsta rap','grime','hardcore hip hop','hip house','hip pop','houston rap','igbo rap','jazz rap','jewish hip hop','kwaito','trap latino','latin hip hop','lo-fi','lo-fi product','memphis hip hop','melodic rap','metropopolis','nerdcore','new jack swing','old school atlanta hip hop','otacore','philly rap','pittsburgh rap','progressive rap','psychedelic rap','queens hip hop','southern hip hop','rap metal','rap rock','trap','trap queen','underground hip hop','west coast rap','west coast trap'],
              'instrumental':['bardcore','instrumental','karaoke','piano cover'],
              'jazz':['adult standards','avant-garde jazz','bebop','cool jazz','contemporary post-bop','dixieland','easy listening','exotica','free jazz','gothic','lounge','jazz','jazz blues','jazz funk','jazz fusion','jazz rock','jazz piano','jazz saxophone','latin jazz','nu jazz','smooth jazz','soul jazz','spiritual jazz','swing','vocal jazz','quiet storm'],
              'new age':['celtic','meditation','new age','new age piano'],
              "children's music":["children's music","preschool children's music",'lullaby','nursery','musica para ninos'],
              'opera':['opera','operetta','classical soprano','italian soprano','classical tenor','italian tenor','classical baritone'],
              'pop':['alternative pop','antiviral pop','art pop','antiviral pop','baroque pop','barbadian pop','bow pop','britpop','c-pop','canadian pop','chamber pop','chalga','classic country pop','classic opm','country pop','chillwave','dance pop','dream pop','electroclash','electropop','europop','experimental pop','french pop','folk pop','folk-pop','futurepop','hip pop','hyperpop','indian pop','indie pop','new wave pop','k-pop','k-pop boy group','latin pop','latin arena pop','urbano latino','metropopolis','mexican pop','new orleans rap','noise pop','pop dance','pop punk','pop rock','pop rap','pop r&b','power pop','progressive pop','post-teen pop','shibuya-kei','sophisti-pop','spanish pop','sunshine pop','swamp pop','synth-pop','minneapolis sound','taiwanese pop','teen pop','uk pop','quiet storm','viral pop'],
              'reggae':['dancehall','reggae','reggae fusion','roots reggae','lovers rock','ragga','reggae rock','reggaeton'],
              'rhythm and blues':['rhythm and blues','r&b','alternative r&b','contemporary r&b','canadian contemporary r&b','funk','soul','boogie','doo-wop','pop r&b','urban contemporary','quiet storm'],
              'rock':['album rock','alternative rock','british invasion','classic rock','christian rock','christian alternative rock','heartland rock','heavy metal','indie rock','irish rock','modern rock','modern country rock','rock-and-roll','punk rock','acid rock','alternative rock','arena rock','art rock','avant-prog','baroque pop','blues rock','britpop','celtic rock','country rock','comedy rock','dance rock','death rock','desert blues','electronic rock','experimental rock','folk rock','funk rock','minneapolis sound','garage rock','glam rock','gothabilly','gothic rock','grunge','hard rock','latin rock','metal','instrumental rock','jazz rock','k-pop girl group','latin alternative','mellow gold','neo-progressive rock','new wave','noise rock','permanent wave','proto-metal','pop rock','post-grunge','post-punk','post-rock','pov: indie','power pop','progressive country','progressive rock','proto-punk','psychedelic rock','pub rock','punk','rap rock','reggae rock','rockabilly','rock en espanol','soft rock','southern rock','space rock','surf music','surf rock','swamp rock','symphonic rock','synth rock','yacht rock'],
              'samba':['bossa nova','pagode'],
              'sound':['environmental','sleep','sound','water'],
              'singer-songwriter':['singer-songwriter'],
              'soundtrack':['soundtrack','theme','british soundtrack','orchestral soundtrack','anime', 'anime score', 'japanese classical', 'japanese soundtrack', 'orchestral soundtrack','japanese vgm','japanese celtic'],
              'soul':['blue-eyed soul','british soul','classic soul','motown','neo soul','pop soul','northern soul','southern soul','progressive soul','psychedelic soul','soca','soul jazz','quiet storm'],
              'workout product':['workout product'],
              'world music':['banda','boleros','bolero','calypso','cha-cha-cha','classic opm','conga','cuban','dance hall','dub','exotica','gamelan','kulintang','igbo highlife','mamob','mariachi','musica mexicana','musica sonorense','norteno','old school dancehall','rocksteady','rumba','salsa','sierreno','reggae','ranchera','ska','traditional','tropical','urbano latino','worldbeat','zouk']
              }

# Replace each subgenre name as their main genre
spotify_merged['genres'].replace({subgenre:genre for genre,subgenres in genre_dict.items() for subgenre in subgenres},inplace=True)''',language='python')
    
    st.markdown("Afterwards, I have extracted the part of dataframe containing the top 25 most appeared genres in the dataset. I have created a frequency table of each element in the genre column, extracted the top 25 most frequently occurred genres, and sliced the rows of the dataframe where any of the top 25 genres are present in the genre column.")
    
    with st.expander("Click to view the code to slice the top 25 genres"):
        st.code('''# Generate the frequency table of genres
genre_count = spotify_merged['genres'].value_counts()
top_25 = pd.DataFrame(genre_count[:25])
# Extract the names of the top 25 genres, stored as the frequency table's index
top_25_genres = top_25.index

# Create a function that determine if a genre is in 'top_25_genres'
def in_top_25(element):
    if element in top_25_genres:
        return True
    else:
        return False

# Use the function to slice the Spotify dataframe & make the top 25 genre data
top_25_column = spotify_merged['genres'].apply(in_top_25)
top_25_genre_data = spotify_merged[top_25_column]
#Finally, save the Spotify dataframe as a copy of top_25_genre_data
spotify = top_25_genre_data.copy()''',language='python')
    
    st.markdown("Here is the original genres column (left) and the cleaned 'genres' column (right):")
    st.dataframe(spotify.loc[:,['artist_genres','genres']])

    
    st.header('Label Cleaning')
    
    st.markdown('Cleaning the music labels involves a more complicated process than cleaning genres. The steps mainly involve manually research the main labels (Sony, Warner, Universal, or independent) that each specific labels are under, web scraping label names, and updating labels by their initial characters (10:22pm, 10k, etc.).')
    st.markdown("Similar to cleaning genres, I have researched the top sublabels in the dataset, organized them into their main labels, and made a new column 'main_label' that converts the sublabels into their main labels.")
    
    with st.expander("Click to view the label cleaning code"):
        st.code('''# Create the lists of sublabels per each main label
sony_subs = ['Sony Music','Sony Classical','Sony Music Latin','Columbia','Columbia Records','Columbia/Legacy','Columbia Nashville Legacy','RCA Records','RCA Records Label','Arista','Arista Records','Arista/Legacy','Epic','Epic/Legacy','Epic/Freebandz','Epic/Freebandz/A1','Jive','RCA/Legacy','RCA','Legacy','Legacy Recordings']
warner_subs = ['Warner Records','Warner Classics','Warner Classics International','300 Entertainment','Atlantic Records','Asylum','Big Beat','Nonesuch','Maverick','Sire','Reprise','Parlophone','Chrysalis','EMI','Harvest','Parlophone (France)','Parlophone UK','Regal','Rhino','Rhino/Warner Records','Roadrunner Records','TK','Roulette','Bearsville','Del-Fi','Atco','Elektra Records','Rhino/Elektra','Naxos','earMUSIC','Generation Now/Atlantic','Priority Records']
universal_subs = ['CM/Republic','Geffen*','Ultra Records, LLC','CAPITOL CATALOG MKT (C92)','eOne Music','Rhino Atlantic','Mercury Studios','Young Money Records, Inc.','Universal-Island Records Ltd.','MCA Nashville','Universal Music, a division of Universal International Music BV','Big Machine Records, LLC','300 Entertainment/Atl','Deutsche Grammophon (DG)','Interscope','Interscope Records','Geffen','Geffen Records','UNI/MOTOWN','MOTOWN','Motown','A&M','Aftermath','CM/Republic','CM','Capitol','Capitol Records','Decca Music Group Ltd.','Def Jam Recordings','Republic','Republic Records','Island Records','Polydor Records','Verve Reissues','UMC (Universal Music Catalogue)','SM Entertainment (often distributed by various majors including Universal)','Musical Freedom','Universal Records','Cash Money Records/Young Money Ent./Universal Rec.','WM Mexico','Universal Music Spain S.L.','Nicki Minaj/Cash Money','Universal Music Mexico','UME - Global Clearing House','Taylor Swift']

# Create the dictionary of labels, with the main label as the key, and the list of sublabels as its values
label_dict = {'Sony':sony_subs, 'Warner':warner_subs, 'Universal':universal_subs}
# Replace each sublabel name as their main label 
spotify.loc[:,'main_label'] = spotify['label'].replace({sub_label:label for label,sub_labels in label_dict.items() for sub_label in sub_labels})''',language='python')
    
    st.markdown("Afterwards, I have used Python's request and BeautifulSoup libraries to scrape all the label names, from the Wikipedia pages about the list of Sony, Warner and Universal music labels. Each label name is displayed as the titles of each list element <li>. For example, here is how I scraped the sublabel names from the Sony Labels page:")
    with st.expander("Click to view the web scraping code"):
        st.code('''# URL of the Wikipedia page
sony_url = "https://en.wikipedia.org/wiki/List_of_Sony_Music_labels"

# Fetch the content of the page
response = requests.get(sony_url)

content = response.content

# Parse the content with BeautifulSoup
soup = BeautifulSoup(content, 'html.parser')

all_ul = soup.find_all('ul')

sony_titles = []

for ul in all_ul:
    # Find all li elements within each ul
    li_elements = ul.find_all('li')
    for li in li_elements:
        # Check if li contains an a tag
        a_tag = li.find('a')
        if a_tag:
            # Extract the title attribute if it exists
            title = a_tag.get('title')
            if title:
                sony_titles.append(title)
        else:
            # Extract the text content of the li element
            text = li.get_text(strip=True)
            sony_titles.append(text)''', language='python')
    
    
    st.markdown("As there are still many scraped sublabels that are unmatched with the sublabels in the DataFrame, I had to create a function 'update_labels' that updates the 'clean_labels' dictionary, which its key is the sublabel and its value is the main label. Then, I implemented a loop that uses the function to convert the label names into their main label based on the first 3-7 characters (like 10:22pm, 10k, 143, atlantic)")
    with st.expander("Click to view the update label function and loop"):
        st.code("""clean_labels = {}

def update_labels(label, string, inplace=False):
    '''
    Use the first 3-7 characters as the input string, to match all instances of the corresponding
    label name with the scraped sublabels that are not yet matched with what's in the dataframe; 
    then, update the clean_label dictionary with the key as the matched label and the value as its main 
    label
    '''
    temp_data_labels = data_labels.str.replace('\(', ' ').str.replace('\)', ' ')
    results = temp_data_labels[temp_data_labels.str.contains(string)]
    main_label = all_scraped_labels_dict[label]
    dict_results = {k:main_label for k in results}
    if inplace:
        clean_labels.update(dict_results)
    if len(results) == 0:
        return dict_results, label
    return dict_results, None

still_unmatched_scraped = []

# Iterate through the rest of unmatched sublabels that are web-scraped
for label in unmatched_scraped_labels[1:]:
    '''
    First, replace the paranthesis characters as it will generate an error when the first 3-7
    characters included an open paranthesis without a closing one; Afterwards, use a loop to extract the 
    string with first 3-7 characters and convert label names into their main label using the string
    '''
    temp_label = label.replace('(', ' ')
    temp_label = temp_label.replace(')', ' ')
    for i in range (3,8):
        match_dict, match_label = update_labels(label, temp_label[:i], inplace=True)
        if match_label is not None:
            still_unmatched_scraped.append([match_label, temp_label])""", language='python')
    
    st.markdown("Finally, here is the cleaned 'main_label' column:")
    st.dataframe(spotify.loc[:,['label','main_label']])
    
    
    
    st.markdown("Each of the musical features (danceability, valence, energy, etc.) needs to be converted to a level variable and assigned to a column. I have set the 'High' level to above or equal to 0.7 (70%), 'Decent' level to 0.5-0.7 (50-70%), and 'Low' level to below 0.5 (50%). I have defined a function in order to do this:")
    func_code = '''def label(element):
    if element >= 0.7:
        return 'High'
    elif element >= 0.5:
        return 'Decent'
    elif element < 0.5:
        return 'Low'
    else:
        return 'Unknown' '''
    st.code(func_code,language='python')
    
    st.markdown("Then the function is applied on each of the musical attribute columns to create new columns for each. For example, the following line shows that the function is applied on the danceability column, to create a 'danceability_level' column to organize the high to low danceability levels:")
    apply_code = '''spotify.loc[:,'danceability_level'] = spotify['danceability'].map(label)'''
    st.code(apply_code,language='python')
    
    
    st.markdown("There are some columns in the dataset, like 'mode' and 'key', their elements are represented as numbers instead of words (such as in 'mode', 'major' is represented as 1, 'minor' is represented as 0). This will create confusions when analyzing the data, so we have to replace the numbers with their corresponding verbal meanings. To do so, we have to convert these variables from numeric to string format, create a dictionary of replacement values, and apply the dictionary to replace values for the variables.")
    replace_code = '''key_dict = {'0.0':'C','1.0':'C#/Db','2.0':'D','3.0':'Eb','4.0':'E','5.0':'F','6.0':'F#/Gb','7.0':'G','8.0':'G#/Ab','9.0':'A','10.0':'Bb','11.0':'B'}
spotify.loc[:,'key'] = spotify['key'].astype(str)
spotify.loc[:,'key'] = spotify.loc[:,'key'].replace(key_dict)
spotify.loc[:,'mode'] = spotify.loc[:,'mode'].astype(str).replace({'1.0':'Major','0.0':'Minor'})'''
    st.code(replace_code,language='python')
    
    
    st.markdown("Finally, here is part of the cleaned dataset (since the original dataset is really large, the last 50 rows are displayed instead):")
    st.dataframe(spotify.tail(50))



if selected=="Exploratory Analysis":
    st.title('Exploratory Analysis')
    st.markdown("In this section, we will explore the types of variables (numeric & categorical) of our dataset, and how those variables influence artist popularity. By visualizing data using various types of graph (scatter, box plot, histogram, etc.), we can have a clear idea of how each variable relates to the overall artist popularity. You are free to choose any variable(s) and display the graph to see how much it relates to artist popularity.")
    
    col3,col4=st.columns([3,5])
    col3.markdown(" ")
    
    col3.header("Box Plot: Numeric Variables vs. Artist Popularity Level")
    
    with st.form("Box plot"):
        y_select = col3.selectbox("Select one numeric variable", num_dict.values(),key=2)
        y_select_new = [k for k,v in num_dict.items() if v == y_select][0]
        submitted=st.form_submit_button("Submit to produce the box plot")
        if submitted:
            fig = px.box(spotify_graph, x='art_pop_level', y=y_select_new, color='art_pop_level', color_discrete_sequence=color_list, category_orders=order_dict, labels=all_cols_dict, hover_data=['track_name','artist_0'], title=f"Artist popularity level vs. {y_select}")
            fig.update_traces(marker_line_width=1)
            fig.update_xaxes(title_text="<b>Artist Popularity Level</b>", title_font_size=14)
            fig.update_yaxes(title_text=f"<b>{y_select}</b>", title_font_size=14)
            fig.update_layout(title_x=0.2, legend_title_font_size=14, legend_title_text="<b>Artist Popularity Level</b>", legend_bordercolor="green", legend_borderwidth=2, hoverlabel_font_size=14)
            fig.for_each_annotation(lambda a: a.update(text=f'<b>{a.text.split("=")[-1]}</b>',font_size=14))
            col4.plotly_chart(fig)
    
    
    col1,col2=st.columns([3,5])
    col1.markdown(" ")
    
    col1.header("Histogram: Distribution of Numeric Variables")
    
    with st.form("Histogram"):
        x_select = col1.selectbox("Select one numeric variable",num_dict.values(),key=1)
        x_select_new = [k for k,v in num_dict.items() if v == x_select][0]
        check1 = col1.checkbox("Specify the number of bins",key=88)
        check2 = col1.checkbox("Normalized histogram",key=66)
        n = 10
        percent = None
        percent_text = 'Count'
        if check1:
            n = col1.number_input('Insert a number', min_value=10, step=10)
        if check2:
            percent = 'percent'
            percent_text = 'Percent (%)'
        submitted=st.form_submit_button("Submit to produce the histogram")
        if submitted:
            fig = px.histogram(spotify_graph, x=x_select_new, nbins=n, color='art_pop_level', color_discrete_sequence=color_list, category_orders=order_dict, labels=all_cols_dict, barmode='group', histnorm=percent, title=f"Artist popularity level vs. {x_select}")
            fig.update_traces(marker_line_width=1)
            fig.update_xaxes(title_text=f"<b>{x_select}</b>", title_font_size=14)
            fig.update_yaxes(title_text=f"<b>{percent_text}</b>", title_font_size=14)
            fig.update_layout(title_x=0.2, legend_title_font_size=14, legend_title_text="<b>Artist Popularity Level</b>", legend_bordercolor="green", legend_borderwidth=2, hoverlabel_font_size=14)
            fig.for_each_annotation(lambda a: a.update(text=f'<b>{a.text.split("=")[-1]}</b>',font_size=14))
            col2.plotly_chart(fig)
    
    
    col7,col8=st.columns([3,5])
    col7.markdown(" ")
    
    col7.header("Bar Chart: Distribution of Categorical Variables")
    
    with st.form("Bar"):
        x_select = col7.selectbox("Select one category variable",cat_dict.values(),key=5)
        x_select_new = [k for k,v in cat_dict.items() if v == x_select][0]
        check = col7.checkbox("Normalized bar chart",key=68)
        percent = None
        percent_text = 'Count'
        if check:
            percent = 'percent'
            percent_text = 'Percent (%)'
        submitted=st.form_submit_button("Submit to produce the bar chart")
        if submitted:
            fig = px.histogram(spotify_graph, x=x_select_new, color='art_pop_level', color_discrete_sequence=color_list, category_orders=order_dict, barmode='group', histnorm=percent, labels=all_cols_dict, title=f"Artist popularity level vs. {x_select}")
            fig.update_traces(marker_line_width=1)
            fig.update_xaxes(title_text=f"<b>{x_select}</b>", title_font_size=14)
            fig.update_yaxes(title_text=f"<b>{percent_text}</b>", title_font_size=14)
            fig.update_xaxes(categoryorder='total descending')
            fig.update_layout(title_x=0.2, legend_title_font_size=14, legend_title_text="<b>Artist Popularity Level</b>", legend_bordercolor="green", legend_borderwidth=2, hoverlabel_font_size=14)
            fig.for_each_annotation(lambda a: a.update(text=f'<b>{a.text.split("=")[-1]}</b>',font_size=14))
            col8.plotly_chart(fig)
    
    
    col9,col10=st.columns([3,5])
    col9.header("Scatter Plot: Numerical Variables Comparison")
    
    with st.form("Scatter"):
        x_select = col9.selectbox("select one variable for the x-axis", num_dict.values(), key=43)
        x_select_new = [k for k,v in num_dict.items() if v == x_select][0]
        y_select = col9.selectbox("select one variable for the y-axis", np.setdiff1d(list(num_dict.values()), x_select), key=44)
        y_select_new = [k for k,v in num_dict.items() if v == y_select][0]
        submitted=st.form_submit_button("Submit to produce the scatterplot")
        if submitted:
            fig = px.scatter(spotify_graph, x=x_select_new, y=y_select_new, color='art_pop_level', color_discrete_sequence=color_list, labels=all_cols_dict, title=f"{x_select} vs. {y_select}")
            fig.update_traces(marker_line_width=1)
            fig.update_xaxes(title_text=f"<b>{x_select}</b>", title_font_size=16)
            fig.update_yaxes(title_text=f"<b>{y_select}</b>", title_font_size=16)
            fig.update_layout(title_x=0.3, legend_title_font_size=14, legend_title_text="<b>Artist Popularity Level</b>", legend_bordercolor="green", legend_borderwidth=2, hoverlabel_font_size=14)
            col10.plotly_chart(fig)
    
    
    col11,col12=st.columns([3,5])
    col11.header("KDE (Kernel Density Estimation) Graph: Probability Density Based on Different Categories")
    
    with st.form("KDE"):
        num_select = col11.selectbox("Select one numeric variable",num_dict.values(), key=130)
        num_select_new = [k for k,v in num_dict.items() if v == num_select][0]
        cat_select = col11.selectbox("Select one category variable",cat_dict.values(), key=131)
        cat_select_new = [k for k,v in cat_dict.items() if v == cat_select][0]
        specific_category = col11.selectbox("Select a specific category", sorted(spotify[cat_select_new].unique()), key=132)
        submitted=st.form_submit_button("Submit to produce the KDE graph")
        if submitted:
            data_popular = spotify[(spotify['art_pop_level']=='Popular') & (spotify[cat_select_new]==specific_category)][num_select_new].dropna()
            data_unpopular = spotify[(spotify['art_pop_level']=='Unpopular') & (spotify[cat_select_new]==specific_category)][num_select_new].dropna()
            fig4=go.Figure()
            if len(data_popular) != 0:
                kde_popular = stats.gaussian_kde(data_popular)
                fig4.add_trace(go.Scatter(x=np.arange(data_popular.min(),data_popular.max()+1), y=kde_popular.evaluate(np.arange(data_popular.min(),data_popular.max()+1)), mode='lines', name='Popular', line_color=color_list[0], hovertemplate="<b>Popular</b><br><br>x=%{x}<br>y=%{y}<extra></extra>"))
            if len(data_unpopular) != 0:
                kde_unpopular = stats.gaussian_kde(data_unpopular)
                fig4.add_trace(go.Scatter(x=np.arange(data_unpopular.min(),data_unpopular.max()+1), y=kde_unpopular.evaluate(np.arange(data_unpopular.min(),data_unpopular.max()+1)), mode='lines', name='Unpopular', line_color=color_list[1], hovertemplate="<b>Unpopular</b><br><br>x=%{x}<br>y=%{y}<extra></extra>"))
            fig4.add_vline(data_popular.mean(), line_color=color_list[0])
            fig4.add_vline(data_unpopular.mean(), line_color=color_list[1])
            fig4.update_yaxes(title_text='<b>Density</b>')
            fig4.update_xaxes(title_text=f'<b>{num_select}</b>')
            fig4.update_layout(width=800, height=500, title_text=f"<b>Kernel Density of {num_select} in {specific_category} {cat_select} with their respective means</b>")
            col12.plotly_chart(fig4)
            col12.markdown(f"Note: the vertical line across the graph indicates the mean of {num_select} in {specific_category} {cat_select}.")
    
    
    
    col13,col14=st.columns([3,5])
    
    col13.header("Sunburst graph: Numeric Variables vs. Genres, Main Label & Artist Popularity Level")
    
    with st.form("Sunburst"):
        value_select = col13.selectbox("Select one numeric variable", num_dict.values(), key=26)
        value_select_new = [k for k,v in num_dict.items() if v == value_select][0]
        path_order = ['art_pop_level','genres','main_label']
        check = col13.checkbox("Check to view label before genre",key=100)
        if check:
            path_order = ['art_pop_level','main_label','genres']
        color_radio = col13.radio("Choose the feature to base the color on",['genres','art_pop_level','main_label'],key=107)
        submitted=st.form_submit_button("Submit to produce the graph")
        if submitted:
            fig = px.sunburst(spotify, path=path_order, values=value_select_new, color=color_radio, color_discrete_sequence=color_list, labels=all_cols_dict, title=f"{value_select} vs. Genres, Main Label & Artist Popularity Level")
            fig.update_traces(textinfo="label+percent parent")
            fig.update_layout(width=600, height=600)
            col14.plotly_chart(fig)
    
    
    
    col5,col6=st.columns([3,5])
    col5.header("Chi-Square")
    col5.markdown("To determine the statistical relationship between two categorical variables, a Chi-square test of independence can be implemented to check whether the two variables are likely related or independent to each other. A larger Chi-square statistic indicates a stronger relation between the two variables, while a smaller Chi-square statistic indicates that they're more likely independent (i.e., weaker relation between them). We will generate a bar plot that shows each selected target variable versus their Chi-square statistics, to see which ones are more strongly related to the selected target variable.")
    
    predictors = st.multiselect("Select predictor variables", cat_dict.values(), key=20, default=None)
    predictors_new = [k for k,v in cat_dict.items() if v in predictors]
    target = st.selectbox("Select Target Variable", np.setdiff1d(list(cat_dict.values()), predictors), key=23)
    target_new = [k for k,v in cat_dict.items() if v == target][0]
    check_log = st.checkbox("Display log scale")
    logy = False
    if check_log:
        logy = True
    submitted=st.button("Submit to generate the graph")
    if submitted:
        chi_explore = [[i] + list(stats.chi2_contingency(pd.crosstab(spotify[i],spotify[target_new]))[0:2]) for i in predictors_new]
        chi_data = pd.DataFrame(chi_explore, columns=['Predictor','Chi-Square Statistic','P-Value'])
        chi_data['Predictor'] = chi_data['Predictor'].replace({k:v for k,v in cat_dict.items()})
        fig = px.bar(chi_data, x='Predictor', y='Chi-Square Statistic', hover_name='Predictor', hover_data='P-Value', color='Predictor', title=f'<b> Chi-Square Value of Predictor Variables and {target}</b>',log_y=logy)
        fig.update_xaxes(categoryorder='total descending')
        col6.plotly_chart(fig)






if selected == "Testing Naive Bayes Predictions":
    st.title("Interactive Naive Bayes Predictions")
    with st.form("Naive Bayes"):
        predictors = st.multiselect("Select multiple variables that you want to include to predict artist popularity", np.setdiff1d(list(num_dict.values())+list(cat_dict.values()), ['Artist Popularity','Artist Popularity Level']), key=20, default=None)
        predictors_new = [k for k,v in num_dict.items() if v in predictors]
        predictors_new += [k for k,v in cat_dict.items() if v in predictors]
        submitted = st.form_submit_button("Submit to view predictions and accuracy")
        if submitted:
            predicted, acc = naive_bayes(spotify, predictors_new)
            st.markdown(f"The final accuracy of Naive Bayes is: {np.round(acc,4)}")
            st.dataframe(predicted)
    st.markdown("Click to view the following code being implemented for my Naive Bayes Classifier if you're interested to understand how it works!")
    with st.expander("View the function"):
        st.code('''def naive_bayes(df,predictors,tv='art_pop_level'):
    popular = df[df[tv]=='Popular'].copy()
    unpopular = df[df[tv]=='Unpopular'].copy()

    num_cols = np.setdiff1d(df.select_dtypes(include='number').columns, ["artist_popularity","duration_ms","released_day","rn","total_tracks"])
    cat_cols = np.setdiff1d(df.select_dtypes(include=['category','object']).columns,["album_id","album_name","artists","artist_genres","artist_id","artist_1","artist_2","artist_3","artist_4","artist_genres","genre_0","genre_1","genre_2","genre_3","genre_4","label","name","release_date","track_id","track_name","analysis_url","uri","type","track_href"])

    pop_prior = df[tv].value_counts(normalize=True).loc['Popular']
    unpop_prior = df[tv].value_counts(normalize=True).loc['Unpopular']
    
    pop_dict = {}
    un_dict = {}

    for c in predictors:
        if c in num_cols:
            pop_dict.update({c:fit_distribution(popular[c]).pdf(df[c])})
            un_dict.update({c:fit_distribution(unpopular[c]).pdf(df[c])})
        elif c in cat_cols:
            p_con = dict(popular[c].value_counts(normalize=True))
            un_con=dict(unpopular[c].value_counts(normalize=True))
            pop_dict[c] = df[c].map(p_con)
            un_dict[c] = df[c].map(un_con)
    
    pop_prob = pd.DataFrame(pop_dict).prod(axis=1) * pop_prior
    unpop_prob = pd.DataFrame(un_dict).prod(axis=1) * unpop_prior

    predicted = (pop_prob > unpop_prob).map({True:'Popular',False:'Unpopular'})
    results = pd.concat([predicted.reset_index(drop=True),df[tv].reset_index(drop=True)],axis=1)
    
    results.columns = ['Predicted','Actual']

    accuracy = np.mean(results['Predicted'] == results['Actual'])
    
    return results, accuracy''',language='python')
    

if selected == "Data Analysis":
    st.title("Data Analysis")
    st.markdown('')
    st.markdown("Now we will identify the factors that will most likely affect artist popularity, by the process of selecting effective predictors for the Naive Bayes model to make the most accurate predictions. Accuracy is crucial to evaluate the Naive Bayes model's performance, and a good prediction model needs a benchmark percentage of accuracy to be evaluated against. In our dataset, the most frequent category of artist popularity level is 'Popular', which takes up 74.57% of the whole data. Hence, a good model must have an accuracy higher than that.")
    st.write(spotify['art_pop_level'].value_counts(normalize=True))
    
    
    st.header("Naive Bayes Accuracy for All Variables")
    col1,col2=st.columns([3,5])
    col1.markdown("First, I have used all the variables as the predictors to my Naive Bayes model. This results in an 81% accuracy, which is ideal as it is higher than the baseline accuracy of 74.57%.")
    col1.markdown("However, in the analysis, more effective predictors should be selected for my model to improve its overall accuracy. If we include the most correlated variables with the target variable (artist popularity level) as the predictors, then it may result in an overall better accuracy.")
    col2.markdown(f"The accuracy of Naive Bayes model: {naive_bayes(spotify,np.setdiff1d(spotify.columns,'art_pop_level'))[1]:.4f}")
    col2.write(naive_bayes(spotify,np.setdiff1d(spotify.columns,'art_pop_level'))[0])
    
    
    st.header("Selecting Category Variables as Predictors")
    st.subheader("Chi-square test")
    
    col3,col4=st.columns([4,5])
    col3.markdown("From the exploratory section, we know that the chi-square test of independence can be used to test whether a variable is independent or correlated with the target variable, artist popularity level. On the right is the DataFrame containing the chi-square statistic and the p-value of each variable, which are the results of the chi-square test.")
    col3.markdown("We can also visualize the chi-square results using bar plot. The higher the chi-squared statistic of a variable, the more likely the variable is closely related with the target variable.")
    
    predictors = np.setdiff1d(list(cat_dict.values()),'Artist Popularity Level')
    predictors_new = [k for k,v in cat_dict.items() if v in predictors]
    target = 'art_pop_level'
    chi_explore = [[i] + list(stats.chi2_contingency(pd.crosstab(spotify[i],spotify[target]))[0:2]) for i in predictors_new]
    chi_data = pd.DataFrame(chi_explore, columns=['Predictor','Chi-Square Statistic','P-Value'])
    chi_data['Predictor'] = chi_data['Predictor'].replace({k:v for k,v in cat_dict.items()})
    chi_data.sort_values('Chi-Square Statistic', ascending=False, ignore_index=True, inplace=True)
    col4.write(chi_data)
    
    col5,col6,col7=st.columns([4,5,6])
    
    col5.markdown('')
    col5.markdown('')
    col5.markdown("From the bar plot, we can clearly see that 'Main Artist' has the highest chi-squared value, followed by 'Genres'. It is more than obvious that if the artists themselves are popular, then the popularity level is of course 'Popular', and vice versa. The artist genres may also be strongly related to artist popularity, especially the popular ones like pop, hip hop, R&B and EDM, as mentioned in previous studies by experts.")
    
    fig = px.bar(chi_data, x='Predictor', y='Chi-Square Statistic', hover_name='Predictor', hover_data='P-Value', color='Predictor', title='<b> Chi-Square Values of Categorical Variables</b>')
    fig.update_layout(title_x=0.5, title_xanchor='center')
    col6.plotly_chart(fig)
    
    st.subheader("Testing the 'most correlated' categorical variables I")
    
    col8,col9=st.columns([4,4])
    
    col8.markdown("However, when both the Main Artist and Genres variables are tested against the model, it results in a low accuracy of 10.67%! This is likely because that the main artist column, 'artist_0', has far too much unique values in the dataset. Since the artist themselves are popular/unpopular already, we will drop this variable and do a chi-square test again.")
    
    col9.write(naive_bayes(spotify,['artist_0','genres'])[0])
    col9.markdown(f"Accuracy: {naive_bayes(spotify,['artist_0','genres'])[1]:.4f}")
    
    
    st.markdown("This time, we will display a log scale on the bar plot, so that the variables with lower chi-squared statistic (less correlated with the target variable) can be seen clearer.")
    predictors = np.setdiff1d(list(cat_dict.values()),['Main Artist', 'Artist Popularity Level'])
    predictors_new = [k for k,v in cat_dict.items() if v in predictors]
    target = 'art_pop_level'
    chi_explore = [[i] + list(stats.chi2_contingency(pd.crosstab(spotify[i],spotify[target]))[0:2]) for i in predictors_new]
    chi_data = pd.DataFrame(chi_explore, columns=['Predictor','Chi-Square Statistic','P-Value'])
    chi_data['Predictor'] = chi_data['Predictor'].replace({k:v for k,v in cat_dict.items()})
    fig = px.bar(chi_data, x='Predictor', y='Chi-Square Statistic', hover_name='Predictor', hover_data='P-Value', color='Predictor', log_y=True, title='<b> Chi-Square Values of Categorical Variables</b>')
    fig.update_layout(title_x=0.5, title_xanchor='center')
    fig.update_xaxes(categoryorder='total descending')
    st.plotly_chart(fig)
    
    st.subheader("Heatmap for correlation between categorical variables")
    
    st.markdown("We have seen that Genres are most relevant to artist popularity level, followed by Level of Energy. To determine the other predictors, we will make a heatmap of correlation between the categorical variables.")
    
    col10,col11=st.columns([4,5])
    col10.markdown("")
    col10.markdown("")
    col10.markdown("")
    
    col10.markdown("The heatmap shows the correlation between each categorical variable in our dataset,in terms of chi-squared values, with a color scale representing the correlation strength between each two vatiables. Yellow indicates a high correlation, green indicates a medium correlation, while cyan indicates a low correlation. Since the Naive Bayes algorithm assumes that all predictor variables are independent, the variables that are strongly related cannot be used as predictors together, such as Acousticness and Energy, or Acousticness and Danceability.")
    col10.markdown("We had Genres and Level of Energy as two of our categorical predictors based on the chi-squared graph. To find other effective predictors, we will examine the ones that are not highly correlated (have dark blue colors) with those two variables.")
    col10.markdown("From the graph, we have found that Main Label and Month of Release are the ones with low correlations with (highly independent towards) both Genres and Level of Energy. We will test the two variables one by one using the Naive Bayes function.")
    
    fig = px.imshow(spotify_chi2, color_continuous_scale=px.colors.sequential.Aggrnyl, text_auto=True, width=900, height=900, labels=all_cols_dict, range_color=[0,300000], title="<b>Correlation of Each Categorical Variable")
    fig.update_traces()
    fig.update_layout(title_x=0.5, title_xanchor='center')
    col11.write(fig)
    
    
    st.subheader("Testing the 'most correlated' categorical variables II")
    
    col15,col16,col17=st.columns([4,4,6])
    
    col15.markdown("Here, we see that Main Label can generate output with a bit higher accuracy than Month of Release, when tested together with Genres and Level of Energy. This shows that the artist's label can also influence artist popularity, as artists under major labels (Sony, Warner, Universal) are more likely to be popular than independent labels, as mentioned in previous studies by experts.")
    
    col16.markdown("Results with Month of Release")
    col16.write(naive_bayes(spotify,['genres','energy_level','release_month'])[0])
    col16.markdown(f"Accuracy: {naive_bayes(spotify,['genres','energy_level','release_month'])[1]:.4f}")
    col17.markdown("Results with Main Label")
    col17.write(naive_bayes(spotify,['genres','energy_level','main_label'])[0])
    col17.markdown(f"Accuracy: {naive_bayes(spotify,['genres','energy_level','main_label'])[1]:.4f}")
    
    st.markdown("")
    st.markdown("Therefore, our final categorical predictors will be Genres, Main Label and Level of Energy.")
    
    
    st.header("Selecting Numeric Variables as Predictors")
    st.subheader("Facet box plots")
    
    st.markdown("Facet plots are useful to directly see the difference in correlations between each variable against the target variable. In this case, we will use box plots as it shows the mean, upper quartile and lower quartile of each variable per category, to compare how much the overall influence each variable has on the target variable.")
    st.markdown("Due to having a huge dataset with a lot of numeric variables, it will be hard to process the graph with all variables in a huge dataset, so I only selected the variables that have significant correlations between it and artist popularity level.")
    

    fig_1 = px.box(melted, x='art_pop_level', y='value', color='art_pop_level', facet_col='variable', facet_col_wrap=3, facet_col_spacing=0.1, width=800, height=800)
    fig_1.for_each_annotation(lambda a: a.update(text=f'<b>{a.text.split("=")[-1]}</b>',font_size=14))
    fig_1.update_yaxes(matches=None,title='',showticklabels=True)
    fig_1.update_xaxes(showticklabels=False,title_text='')
    fig_1.add_annotation(x=-0.05,y=0.4,text="<b>Value of Each Variable</b>", textangle=-90, xref="paper", yref="paper",font_size=14)
    st.plotly_chart(fig_1)
    
    
    st.subheader("Testing the 'most correlated' numeric variables I")
    
    col18,col19=st.columns([4,4])
    
    col18.markdown("We have found that for all the variables above, the popular artists generally have greater mean & quartiles than unpopular artists, indicating a significant correlation. As a result, we have a 92.17% accuracy, even higher than the accuracy with the category variables!")
    col18.markdown("However, if we select carefully the specific predictors, then we are likely to generate an even higher accuracy that optimizes the model's performance.")
    
    col19.write(naive_bayes(spotify, ['danceability', 'energy', 'liveness', 'album_popularity', 'followers', 'track_popularity'])[0])
    col19.markdown(f"Accuracy: {naive_bayes(spotify, ['danceability', 'energy', 'liveness', 'album_popularity', 'followers', 'track_popularity'])[1]:.4f}")
    
    
    st.subheader("Heatmap for correlation between numerical variables")
    
    col20,col21,col22=st.columns([4,5,6])
    col20.markdown("")
    col20.markdown("")
    col20.markdown("")
    
    col20.markdown("Now we have another heatmap for the numerical variables, to measure the strength of correlations between each two variables using the correlation coefficient. In this case, the correlation coefficient is between -1 and 1, where a positive number indicates a positive correlation, and a negative number indicates a negative correlation.")
    col20.markdown("From the graphs above, we saw that Album Popularity, Number of Followers and Track Popularity (which are all popularity factors) have a significant impact on artist popularity, that popular artists obviously have popular albums & tracks, as well as greater number of followers, than unpopular artists. From the heatmap, we will find the variables that are more 'independent' and less correlated to these three variables to be used as predictors.")
    col20.markdown("We see that Danceability and Liveness has the overall least correlation with the popularity variables, so we will test the two variables one by one again, using the Naive Bayes function.")
    
    corr_df = spotify.rename(columns=all_dict)[list(highly_correlated_metrics) + list(highly_correlated_others)].corr().round(4)
    fig = px.imshow(corr_df, color_continuous_scale=px.colors.sequential.Aggrnyl, text_auto=True, width=900, height=900, title="<b>Correlation of Each Numeric Variable")
    fig.update_traces()
    fig.update_layout(title_x=0.5, title_xanchor='center')
    col21.plotly_chart(fig)
    
    
    st.subheader("Testing the 'most correlated' numeric variables II")
    
    col22,col23,col24=st.columns([4,4,6])
    
    col22.markdown("Here, we see that Danceability can generate output with a bit higher accuracy than Liveness, when tested together with Album & Track Popularity and Number of Followers. In this case, Danceability is more likely to influence artist popularity than other musical attributes.")
    col22.markdown("However, if I select Danceability as one of my final predictors, it is inconsistent with the predictor of Level of Energy, which was being previously selected as one of the final category predictors. It seems that despite more popular artists tend to have more energetic and danceable tracks (shown in the facet plots), the music attributes like danceability and energy still doesn't contribute much to the artist's popularity and success. Therefore, the musical attribute variables will not be used as the final predictors for the model.")
    
    col23.markdown("Results with Danceability")
    col23.write(naive_bayes(spotify,['album_popularity', 'followers', 'track_popularity', 'danceability'])[0])
    col23.markdown(f"Accuracy: {naive_bayes(spotify,['album_popularity', 'followers', 'track_popularity', 'danceability'])[1]:.4f}")
    col24.markdown("Results with Liveness")
    col24.write(naive_bayes(spotify,['album_popularity', 'followers', 'track_popularity', 'liveness'])[0])
    col24.markdown(f"Accuracy: {naive_bayes(spotify,['album_popularity', 'followers', 'track_popularity', 'liveness'])[1]:.4f}")
    
    
    st.header("Final Predictors")
    
    st.markdown("Through the experimentation and testing, the final predictors that I will use are: Genres, Main Label, Album Popularity, Followers, and Track Popularity.")
    st.markdown("Using these variables as predictors for the model generates a result with an accuracy of 92.3%, far surpassed the benchmark rate of accuracy (74.57%)! The model's outcome is overall successful, as it uses the most effective predictors to determine the artist's popularity based on the information within a huge dataset.")
    
    st.write(naive_bayes(spotify, ['genres', 'main_label', 'album_popularity', 'followers', 'track_popularity'])[0])
    st.markdown(f"Accuracy: {naive_bayes(spotify, ['genres', 'main_label', 'album_popularity', 'followers', 'track_popularity'])[1]:.4f}")




final_num_predictors = ['Album Popularity','Number of Followers','Track Popularity']
num_predictors_df = pd.melt(spotify.rename(columns=all_cols_dict),id_vars=['Artist Popularity Level'],value_vars=final_num_predictors)

spotify_popular = spotify[spotify['art_pop_level'] == 'Popular'].copy()
spotify_unpopular = spotify[spotify['art_pop_level'] == 'Unpopular'].copy()

pop_genre_count = spotify_popular['genres'].value_counts()
pop_top_5 = pd.DataFrame(pop_genre_count[:5]).index

def in_pop_top_5(element):
    if element in pop_top_5:
        return True
    else:
        return False

pop_top_5_column = spotify_popular['genres'].apply(in_pop_top_5)
pop_top_5_data = spotify_popular[pop_top_5_column]

unpop_genre_count = spotify_unpopular['genres'].value_counts()
unpop_top_5 = pd.DataFrame(unpop_genre_count[:5]).index

def in_unpop_top_5(element):
    if element in unpop_top_5:
        return True
    else:
        return False

unpop_top_5_column = spotify_unpopular['genres'].apply(in_unpop_top_5)
unpop_top_5_data = spotify_unpopular[unpop_top_5_column]


color_genre_map = {'hip hop': color_list[0],
                   'rock': color_list[1],
                   'pop': color_list[2],
                   'classical music': color_list[3],
                   'rhythm and blues': color_list[4],
                   'instrumental': color_list[7],
                   'world music': color_list[6]}



if selected=="Conclusion":
    st.title('Conclusion')
    st.markdown("We have seen that Genres, Main Label, Album Popularity, Followers and Track Popularity creates the highest accuracy of predicting artist popularity, according to the Naive Bayes model. Therefore, the popularity factors like number of followers, album & track popularity can all directly influence artist popularity, while the musical genre and the label that the artist is under is also most likely influence the artist's overall popularity.")
    st.markdown("For the categorical predictors of Genres and Main Label, they contribute to the highest accuracy because of their independence to each other. There are more popular artists with the genres of pop, hip hop, rock, and R&B (rhythm and blues) than the unpopular artists. For the labels, both popular and unpopular artists are mostly under Sony, but a greater portion of unpopular artists are in Warner & Universal than popular artists.")
    
    col1,col2=st.columns([4,5])
    fig1 = px.histogram(pop_top_5_data, x='genres', color='genres', labels=all_cols_dict, width=400, color_discrete_map=color_genre_map, title="Popular Artists' Top 5 Genres")
    fig1.update_traces(marker_line_width=1)
    fig1.update_xaxes(title_text="<b>Genres</b>", title_font_size=14)
    fig1.update_yaxes(title_text="<b>Count</b>", title_font_size=14)
    fig1.update_xaxes(categoryorder='total descending')
    fig1.update_layout(title_x=0.5, title_xanchor='center')
    col1.plotly_chart(fig1)
    
    fig2 = px.histogram(unpop_top_5_data, x='genres', color='genres', labels=all_cols_dict, width=400, color_discrete_map=color_genre_map, title="Unpopular Artists' Top 5 Genres")
    fig2.update_traces(marker_line_width=1)
    fig2.update_xaxes(title_text="<b>Genres</b>", title_font_size=14)
    fig2.update_yaxes(title_text="<b>Count</b>", title_font_size=14)
    fig2.update_xaxes(categoryorder='total descending')
    fig2.update_layout(title_x=0.5, title_xanchor='center')
    col2.plotly_chart(fig2)
    
    fig = px.histogram(spotify, x='main_label', color='main_label', labels=all_cols_dict, facet_col='art_pop_level', width=1000, title="Popular vs. Unpopular Artists' Main Labels")
    fig.update_xaxes(title_text="<b>Main Labels</b>", title_font_size=14)
    fig.update_yaxes(title_text="<b>Count</b>", title_font_size=14)
    fig.update_yaxes(matches=None)
    fig.update_layout(title_x=0.5, title_xanchor='center')
    st.plotly_chart(fig)
    
    
    st.markdown("For the numeric predictors (Album/Track Popularity and Followers), they result in the highest accuracy because of the large difference in means between the popular and unpopular categories. It is obvious that more popular artists tend to gain much more album/track popularity and followers than unpopular artists.")
    
    fig = px.box(num_predictors_df, color='Artist Popularity Level', y='value', facet_col='variable', color_discrete_sequence=color_list, width=1000)
    fig.for_each_annotation(lambda a: a.update(text=f'<b>{a.text.split("=")[-1]}</b>',font_size=14))
    fig.update_yaxes(matches=None,title='',showticklabels=True)
    fig.update_xaxes(showticklabels=False,title_text='')
    fig.add_annotation(x=-0.05,y=0.4,text="<b>Value of Each Variable</b>", textangle=-90, xref="paper", yref="paper",font_size=14)
    st.plotly_chart(fig)

    
    st.markdown("Therefore, popular artists tend to have high album & track popularity, higher number of followers, often engage in genres of hip hop, rock & pop, and they are mostly under Sony and Universal labels.")



if selected=="Bibliography":
    st.title("Bibliography")
    st.markdown("The dataset is downloaded from https://www.kaggle.com/datasets/tonygordonjr/spotify-dataset-2023?resource=download&select=spotify_tracks_data_2023.csv, date accessed, 2024-8-1")
    st.markdown("[1] About Spotify, https://newsroom.spotify.com/company-info/, date accessed, 2024-8-1")
    st.markdown("[2] [3] Spotify Popularity Index: A Little Secret to Help You Leverage the Algorithm, https://www.loudlab.org/blog/spotify-popularity-leverage-algorithm/, date accessed, 2024-8-1")
    st.markdown("[4] How we generate popular tracks, https://support.spotify.com/us/artists/article/how-we-generate-popular-tracks/, date accessed, 2024-8-1")
    st.markdown("[5] Data analysis with spotify api, Erick Lopez, Dec 27, 2023, https://medium.com/@eelopez088/data-analysis-with-spotify-api-a1507f48e9b0")
    st.markdown("[6] The most popular music genres in 2023 ranked through Spotify stats, Nov 25, 2023, https://blog.push.fm/12277/popular-music-genres-2023-ranked-spotify-stats/")
    st.markdown("[7] The Secret Sauce Behind Spotify Recommendations, Software Surplus, Jul 29, 2022, https://medium.com/@software.surplus/the-secret-sauce-behind-spotify-recommendations-74a2b3c3642e")
    st.markdown("[8] Rising to the Top: What makes an artist popular on Spotify?, Ilias Paraskevopoulos, Dec 25, 2020, https://iliasparask1.medium.com/rising-to-the-top-drivers-of-spotify-artist-popularity-4ad9be9532d5")
    st.markdown("[9] Independent label vs The Major labels. Pros and Cons!, Peter Moore, Aug 24, 2020, https://medium.com/the-entertainment-engine/independent-label-vs-the-major-labels-pros-and-cons-cf2b6f78e373")


