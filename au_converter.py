import os
import sys
import audiotools
import pydub
from pydub import AudioSegment


if __name__ == '__main__':
    pass
    f = audiotools.open('/media/files/musicsamples/Go.mp3')
    f.convert(target_path='/media/files/musicsamples/Go_new.au', target_class=audiotools.AuAudio)