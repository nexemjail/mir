#import Tkinter as tkinter
from Tkinter import *
import tkFileDialog
from classifier import get_prediction_vector
import time
import vlc
from helper import get_genre_unmapper
from path_loader import scan_folders_for_au_files
from classifier import  load_svm
from collections import defaultdict
from random import shuffle
import recommender

global files_by_genre_dict, playlist, current_song_genre, sound, classifier, genre_unmapper
global user_prefs, songs_played


def quit(ev):
    global root
    root.destroy()


def get_n_in_genre(genre_dict, genre, N):
    len_avialable = len(genre_dict[genre])
    if len_avialable < N:
        N = len_avialable
    part = genre_dict[genre][:N]
    rest = genre_dict[genre][N:]
    return part, rest


def get_n_in_all_genres(genre_dict, N):
    new_genre_dict = {}
    for genre_name in genre_dict.keys():
        new_genre_dict[genre_name], genre_dict[genre_name] \
            = get_n_in_genre(genre_dict, genre_name, N)
    return new_genre_dict


def get_paths(genre_dict):
    playlist = []
    for (genre_name, path_list) in genre_dict.iteritems():
        playlist.extend(genre_dict[genre_name])
    shuffle(playlist)
    return playlist


def reset_text_view():
    textbox.delete(1.0, END)
    for element in playlist:
        textbox.insert(END, "{0}\n".format(element))


def extend_playlist(extra_paths):
    global playlist
    shuffle(extra_paths)
    playlist.extend(extra_paths)
    reset_text_view()
    #reset_text_view()


def sort_user_prefs():
    global user_prefs
    new_prefs  = defaultdict(int)
    for (k,v) in user_prefs.items():
        if v > 0:
            new_prefs[k] = v
    user_prefs = new_prefs

def get_recommendation():
    #TODO: put it here!
    global files_by_genre_dict,user_prefs
    sort_user_prefs()
    recommendation_tuples = list(recommender.recommend(user_prefs, amount=5).items())
    recommendations = []
    for (genre,amount) in recommendation_tuples:
        part, files_by_genre_dict[genre] = \
            get_n_in_genre(files_by_genre_dict,genre, amount)
        recommendations.extend(part)
    return recommendations


def load_folder(ev):
    global files_by_genre_dict, playlist, classifier
    fn = tkFileDialog.askdirectory(parent =root, mustexist = True,
                                   initialdir = '/media/files/musicsamples/genres')
    files_by_genre_dict = scan_folders_for_au_files(fn)
    genre_dict = get_n_in_all_genres(files_by_genre_dict, 1)
    playlist = get_paths(genre_dict)
    classifier = load_svm()
    reset_text_view()


def start_playing(ev):
    global sound, current_song_genre,playlist
    if sound is None:
        sound = vlc.MediaPlayer(playlist[0])
        sound.play()
        predict(playlist[0],classifier)
    else:
        sound.pause()


def get_next_path_or_none():
    global playlist
    if len(playlist) > 1:
        playlist = playlist[1:]
        reset_text_view()
        return playlist[0]
    return None


def predict(path, classifier, duration = 20, offset = 10):
    global current_song_genre
    current_song_genre = None
    t = time.time()
    pv = get_prediction_vector(path, duration = duration,offset = offset, half_part_length=0.03)
    prediction = classifier.predict(pv)[0]
    current_song_genre = genre_unmapper[prediction]
    print time.time() - t, ' took to predict ', current_song_genre
    return current_song_genre


def next_song():
    print user_prefs
    global sound, playlist, songs_played
    if sound is not None:
        sound.stop()
        path = get_next_path_or_none()
        if path is not None:
            songs_played +=1
            if songs_played % 3 == 0:
                recommendation = get_recommendation()
                extend_playlist(recommendation)
            sound = vlc.MediaPlayer(path)
            sound.play()
            predict(path, classifier,)


def like(event):
    global user_prefs
    if current_song_genre is not None:
        user_prefs[current_song_genre] += 1
    next_song()


def dislike(event):
    global user_prefs
    if current_song_genre is not None:
        user_prefs[current_song_genre] -= 1
    next_song()

if __name__ == "__main__":
    # This is it
    global genre_unmapper, classifier, user_prefs, sound, current_song_genre, songs_played
    sound = None
    current_song_genre = None
    genre_unmapper = get_genre_unmapper()
    classifier = load_svm()
    user_prefs = defaultdict(int)
    songs_played = 0

    root = Tk()
    panelFrame = Frame(root, height = 60, bg = 'gray')
    textFrame = Frame(root, height = 340, width = 600)

    panelFrame.pack(side = 'top', fill = 'x')
    textFrame.pack(side = 'bottom', fill = 'both', expand = 1)

    textbox = Text(textFrame, font='Arial 14', wrap='word')
    scrollbar = Scrollbar(textFrame)

    scrollbar['command'] = textbox.yview
    textbox['yscrollcommand'] = scrollbar.set

    textbox.pack(side = 'left', fill = 'both', expand = 1)
    scrollbar.pack(side = 'right', fill = 'y')

    play_button = Button(panelFrame, text='Play/Pause')
    loadBtn = Button(panelFrame, text = 'Choose folder to play files')
    quitBtn = Button(panelFrame, text = 'Quit')
    button_like = Button(panelFrame, text = 'Like it!')
    button_dislike =Button(panelFrame, text = 'Dislike!')
    loadBtn.bind("<Button-1>", load_folder)
    quitBtn.bind("<Button-1>", quit)
    play_button.bind('<Button-1>', start_playing)
    button_like.bind('<Button-1>', like)
    button_dislike.bind('<Button-1>', dislike)

    loadBtn.place(x = 10, y = 10)
    quitBtn.place(x = 200, y = 10)
    play_button.place(x = 250, y= 10)
    button_like.place(x = 450, y = 10)
    button_dislike.place(x = 550, y = 10)

    root.mainloop()
