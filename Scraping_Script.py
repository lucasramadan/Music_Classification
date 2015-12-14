__author__ = 'Lucas Ramadan'

import requests
from pprint import pprint
import numpy.random as random
import pandas as pd
import requests
import time
import io

# starting with a predefined list of artists to work with
artist_list = [
                # {'genre': 'rap', 'name': 'Mos Def'},
                # {'genre':'rap', 'name': 'Young Jeezy'},
                # {'genre':'rap', 'name': 'Eminem'},
                # {'genre': 'rap', 'name': 'Kendrick Lamar'},
                # {'genre': 'rap', 'name': 'Snoop Dogg'},
                # {'genre': 'rap', 'name': 'Nas'},
                # {'genre': 'rap', 'name': 'Common'},
                # {'genre': 'rap', 'name': 'Kanye West'},
                # {'genre': 'rap', 'name': '2 Chainz'},
                # {'genre': 'rap', 'name': 'Talib Kweli'},
                # {'genre': 'rock', 'name': 'The Who'},
                # {'genre': 'rock', 'name': 'Nirvana'},
                # {'genre': 'rock', 'name': 'Led Zeppelin'},
                # {'genre': 'rock', 'name': 'Aerosmith'},
                # {'genre': 'rock', 'name': 'ACDC'},
                # {'genre': 'rock', 'name': 'Queen'},
                {'genre': 'rock', 'name': 'Pink Floyd'},
                {'genre': 'rock', 'name': 'The Beatles'},
                {'genre': 'rock', 'name': "Guns N' Roses"},
                {'genre': 'rock', 'name': 'Rolling Stones'}
               ]

def get_lyrics(artist_list):

    """
    This function will generate the lyrics for all songs, given a list
    of dictionaries, with artist and genre values using Genius API, and
    write the songs to text files for later analysis.

    INPUT: list of dictionaries - artist_list
    OUTPUT: directory (data/) with songs as .txt files
    """

    # set up a few required links
    ai_link = u'http://genius-api.com/api/artistInfo'
    lyrics_link = u'http://genius-api.com/api/lyricsInfo'

    # this will first generate the list of songs for each artist
    for artist_number, artist_dict in enumerate(artist_list):

        # store the genre and artist name
        genre = artist_dict['genre']
        artist = artist_dict['name']

        # delay random amount of seconds, to try to bypass API threshold
        time.sleep(random.randint(0,5))

        # get the data for the given artist
        a_data = requests.post(ai_link, data = artist_dict).json()

        # we want to get song names and links
        songs = [s['name'] for s in a_data['songs']]
        links = [s['link'] for s in a_data['songs']]

        print
        print songs
        print

        # now we want to iterate over the song, link pairs
        for song_number, (song, link) in enumerate(zip(songs, links)):

            # need to clean the link for use
            if genre == 'rap':
                link = link.replace(u'http://rapgenius.com', '')

            if genre == 'rock':
                link = link.replace(u'http://rock.rapgenius.com', '')

            # generate the song_dict for the request
            song_dict = {u'genre': genre, u'link': link}

            # delay random amount of seconds, to try to bypass API threshold
            time.sleep(random.randint(0,5))

            # get the song data
            song_data = requests.post(lyrics_link, data = song_dict).json()

            # here is where we will store the lyrics to a song
            lyrics = u""

            # now we go through the song_data looking for lyrics
            try:
                for section in song_data['lyrics']['sections']:
                    for verse in section['verses']:
                        try:
                            # build our lyrics
                            lyrics += verse['content']
                            # add space for correct formatting
                            lyrics += u' '
                        except:
                            # show us the content that didn't work
                            pprint(verse)
                            print
                            pass

            # sometimes a song doesn't even have lyrics, so we just skip it
            except:
                pprint(song_data)

            print song

            # quick song name altering for filepath simplicity
            if song.find(' (Ft.') != -1:
                song_name = song[0: song.find(' (Ft.')+1]
            else:
                song_name = song

            print song_name

            # make the filename -- some weird unicode tags are messing it up
            # lots of trial and error here
            try:
                song_filename = 'data/{0} - {1}.txt'.format(artist,
                                    song_name.replace('/', '_'))

            except:
                try:
                    song_filename = 'data/{0} - {1}.txt'.format(artist,
                                        song_name.replace(u'\u200b', ''))
                except:
                    song_filename = 'data/{0} - {1}.txt'.format(artist,
                                        song_name.replace(u'\u2019', ''))

            print song_filename

            # write the lyrics to a file --- need io.open for encoding utf-8
            with io.open(song_filename, "w", encoding='utf-8') as song_file:
                song_file.write(lyrics)

            # display some helpful information when running script
            print "finished %s by %s" % (song, artist)
            print "artist number %d out of %d" % (artist_number+1,
                                                    len(artist_list))
            print "song %d out of %d" % (song_number+1, len(songs))
            print

if __name__ == "__main__":
    get_lyrics(artist_list)
