#import Tkinter as tkinter
from Tkinter import *
import tkFileDialog
import os
import sys
from classifier import convert
from classifier import get_prediction_vector, \
    load_classifiers,predict_genre, get_genre_unmapper
from collections import defaultdict
import time
import numpy as np
import vlc


current_index = 0
files_list = []
sound = None
classifiers = None
predicted_genres = defaultdict(int)
calculating_thread = None
user_intrests = defaultdict(int)
current_song_genre = None


def Quit(ev):
    global root
    root.destroy()


def LoadFolder(ev):
    fn = tkFileDialog.askdirectory(parent =root, mustexist = True,
                                   initialdir = '/media/files/musicsamples')
    global files_list
    for file_name in os.listdir(fn):
        if file_name.endswith('.au'):
            full_path = os.path.join(fn, file_name)
            files_list.append(full_path)
            textbox.insert(END, full_path + '\n')


def init_and_start(path):
    sound = vlc.MediaPlayer(path)
    sound.play()
    return sound

def get_user_rating():
    user_intrests_list = list(user_intrests.items())
    l = sorted(user_intrests_list,key = lambda x: x[1], reverse=True)
    l = [el[0] for el in l if el[1] >= 0]
    print l
    return l


def inc(index):
    index += 1
    if index >= len(files_list):
        global sound
        sound.stop()
        textbox.delete(1.0, END)
        rating = get_user_rating()
        textbox.insert(END, str(rating))
    return index

def predict_current_song_genre(index):
    global predicted_genres, current_song_genre
    t = time.time()
    prediction_vector = get_prediction_vector(files_list[index], duration=20.0, offset=30)
    print 'extracting_features_took ', time.time() - t
    genre_unmapper = get_genre_unmapper()
    for c in xrange(1):
        genre = predict_genre(classifiers[0], prediction_vector, genre_unmapper)
        current_song_genre = genre
        print genre


def StartPlaying(ev):
    global current_index, sound, classifiers, predicted_genres, current_song_genre
    if classifiers is None:
        classifiers = load_classifiers()
    next_song()


def next_song():
    global sound, current_index
    if len(files_list) < 1 or current_index >= len(files_list):
        print 'no files to play'
        return
    if sound is None:
        sound = init_and_start(files_list[current_index])
        predict_current_song_genre(current_index)
        current_index = inc(current_index)
    elif sound.is_playing():
        sound.stop()
        sound = init_and_start(files_list[current_index])
        predict_current_song_genre(current_index)
        current_index = inc(current_index)
    get_user_rating()


def Like(event):
    global user_intrests
    if current_song_genre is not None:
        user_intrests[current_song_genre] += 1
        print user_intrests
        next_song()


def Dislike(event):
    global user_intrests
    if current_song_genre is not None:
        user_intrests[current_song_genre] -= 1
        print user_intrests
        next_song()


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

loadBtn.bind("<Button-1>", LoadFolder)
quitBtn.bind("<Button-1>", Quit)
play_button.bind('<Button-1>', StartPlaying)
button_like.bind('<Button-1>', Like)
button_dislike.bind('<Button-1>', Dislike)


loadBtn.place(x = 10, y = 10)
quitBtn.place(x = 200, y = 10)
play_button.place(x = 250, y= 10)
button_like.place(x = 450, y = 10)
button_dislike.place(x = 550, y = 10)

root.mainloop()
