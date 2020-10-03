"""Visualization tools."""
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import shutil
import math
import os
from PIL import Image, ImageDraw

def save_video(foldername, songname, songlen, num_steps, output):
    """Make video from given frames. Add audio appropriately."""
    num_steps_by_len = num_steps / songlen
    p = subprocess.Popen(['ffmpeg', '-f', 'image2', '-r', str(num_steps_by_len), '-i', '%d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', 'movie.mp4'], cwd=foldername)
    p.wait()

    p = subprocess.Popen(['ffmpeg', '-i', '../audio_files/WeWillRockYou.mp3', '-i', 'movie.mp4','-c', 'copy', '-map', '0:a', '-map', '1:v', output], cwd=foldername)
    p.wait()




def save_matrices(music_matrix, dance_matrix, duration):
    """Save music and dance matrices."""
    d = int(duration) + 1

    plt.tick_params(labelsize=22)
    plt.xticks(np.arange(0, music_matrix.shape[0], music_matrix.shape[0] / duration), np.arange(0, d, 1))
    plt.yticks(np.arange(0, music_matrix.shape[0], music_matrix.shape[0] / duration), np.arange(0, d, 1))
    plt.imshow(music_matrix, cmap='gray')
    plt.savefig('music.png', bbox_inches='tight')
    plt.close()

    plt.tick_params(labelsize=22)
    plt.xticks(np.arange(0, music_matrix.shape[0], music_matrix.shape[0] / duration), np.arange(0, d, 1))
    plt.yticks(np.arange(0, music_matrix.shape[0], music_matrix.shape[0] / duration), np.arange(0, d, 1))
    plt.imshow(dance_matrix, cmap='gray')
    plt.savefig('dance.png', bbox_inches='tight')
    plt.close()


def save_dance(states, visfolder, songname, duration, num_steps):
    """Save dance."""
    # Make folder if not already exists
    if not os.path.exists('./plots/'):
        os.makedirs('plots/')

    # Delete old items
    print("Starting file deletions")
    for item in os.listdir('./plots/'):
        delfile = os.path.join('./plots/', item)
        os.remove(delfile)
    print("File deletions complete")

    # Create dance video
    print("Creating dance video frames")
    c = 0
    for i, state in enumerate(states):
        # ****************** stick figure agent ******************
        shutil.copy(visfolder + "/" + str(state+1) + '.png', 'plots/' + str(i+1) + '.png')
        c += 1

    # Save video
    save_video('./plots', songname, duration, num_steps, songname + '.mov')


def save_circles_dance(voice_states, instrumental_states, visfolder, songname, duration, num_steps):
    print("num steps: " + str(num_steps))
    # Make folder if not already exists
    if not os.path.exists('./circle_plots/'):
        os.makedirs('circle_plots/')

    # Delete old items
    print("Starting file deletions")
    for item in os.listdir('./plots/'):
        delfile = os.path.join('./plots/', item)
        os.remove(delfile)
    print("File deletions complete")

     # Create dance video
    print("Creating dance video frames")
    c = 0
    for i in range(num_steps):
        # ****************** circle creation ******************
        image = Image.new('RGBA', (600, 600), (0,0,0))
        draw = ImageDraw.Draw(image)
        x = 600 - ((voice_states[i]/20) * 600)
        y = 300
        diameter = ((instrumental_states[i]/20) * 300)
        x1 = x - (math.sqrt(.5) * diameter)/2
        y1  = y - (math.sqrt(.5) * diameter)/2
        x2 = x + (math.sqrt(.5) * diameter)/2
        y2 = y + (math.sqrt(.5) * diameter)/2
        # print(str(x1) + " " + str(y1) + " " + str(x2) + " "+ str(y2))
        draw.ellipse((x1, y1, x2, y2), fill = (128, 128, 128), outline =(0, 0, 0))
        image.save("plots/" + str(i+1) + '.png')
        c += 1

    # Save video
    save_video('./plots', songname, duration, num_steps, songname + '.mov')