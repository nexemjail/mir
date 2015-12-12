#import Tkinter as tkinter
from Tkinter import *
import tkFileDialog
import os
import sys
from process_parallelizer import convert_with_processes
from classifier import get_prediction_vector, \
    load_classifiers,predict_genre
from collections import defaultdict
import time
import numpy as np
import vlc
import threading
import recommender
from helper import get_genre_unmapper
from threading import Thread
import os
from path_loader import scan_folders_for_au_files
from classifier import predict_genre, load_svm
from collections import defaultdict

global files_by_genre_dict, playlist, current_song_genre, sound, classifier, genre_unmapper
global user_prefs


def quit(ev):
    global root
    root.destroy()


def get_first_N_by_genre(genre_dict, N):
    # we trunctate input dict
    new_genre_dict = {}
    for (genre_name,path_list) in genre_dict.iteritems():
        len_required = N
        avialable_len = len(genre_dict[genre_name])
        if avialable_len < len_required :
            len_required = avialable_len
        new_genre_dict[genre_name] = genre_dict[genre_name][:len_required]
        genre_dict[genre_name] = genre_dict[genre_name][len_required:]
    return new_genre_dict


def get_paths(genre_dict):
    playlist = []
    for (genre_name, path_list) in genre_dict.iteritems():
        playlist.extend(genre_dict[genre_name])
    return playlist


def reset_text_view():
    textbox.delete(1.0, END)
    for element in playlist:
        textbox.insert(END, "{0}\n".format(element))


def load_folder(ev):
    global files_by_genre_dict, playlist, classifier
    fn = tkFileDialog.askdirectory(parent =root, mustexist = True,
                                   initialdir = '/media/files/musicsamples/genres')
    files_by_genre_dict = scan_folders_for_au_files(fn)
    song_by_genre_dict = get_first_N_by_genre(files_by_genre_dict,2)
    playlist = get_paths(song_by_genre_dict)
    classifier = load_svm()
    reset_text_view()


def start_playing(ev):
    global sound, current_song_genre
    if sound is None:
        sound = vlc.MediaPlayer(playlist[0])
        sound.play()
        predict(playlist[0],classifier)


def get_next_path_or_none():
    global playlist
    if len(playlist) > 1:
        playlist = playlist[1:]
        reset_text_view()
        return playlist[0]
    return None


def predict(path, classifier, duration = 3, offset = 10):
    global current_song_genre
    t = time.time()
    pv = get_prediction_vector(path, duration = duration,offset = offset)
    prediction = classifier.predict(pv)[0]
    current_song_genre = genre_unmapper[prediction]
    print time.time() - t, ' took to predict ', current_song_genre
    return current_song_genre


def next_song():
    print  user_prefs
    global sound, playlist
    if sound is not None:
        sound.stop()
        path = get_next_path_or_none()
        if path is not None:
            sound = vlc.MediaPlayer(path)
            sound.play()
            predict(path, classifier,)


def like(event):
    if current_song_genre is not None:
        user_prefs[current_song_genre] += 1
    next_song()


def dislike(event):
    if current_song_genre is not None:
        user_prefs[current_song_genre] -= 1
    next_song()


if __name__ == "__main__":
    # This is it
    global genre_unmapper, classifier, user_prefs, sound, current_song_genre
    sound = None
    current_song_genre = None
    genre_unmapper = get_genre_unmapper()
    classifier = load_svm()
    user_prefs = defaultdict(int)

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

    play_button = Button(panelFrame, text='Start playing')
    loadBtn = Button(panelFrame, text = 'Choose folder to play files')
    quitBtn = Button(panelFrame, text = 'Quit')
    button_like = Button(panelFrame, text = 'Like it!')
    button_dislike = Button(panelFrame, text = 'Dislike!')

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
