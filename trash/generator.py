import random as r
import os.path
import string


def generate_numeric_file(filename, filesize):
    if os.path.exists(filename):
        print "file already exists"
    else:
        with open(filename, 'w') as f:
            f.writelines([str(r.randint(-20, 20)) + "\n" for _ in xrange(filesize)])


def generate_text_file(filename, filesize):
    with open(filename,'w') as f:
        letters = string.letters
        for _ in xrange(filesize):
            if _ % 20 == 0:
                f.write(" ")
                continue
            if r.randint(1,1000) == 51:
                f.write("\n")
            else:
                f.write(r.choice(letters))


