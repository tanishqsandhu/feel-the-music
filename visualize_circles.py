"""Script to generate all dances for a song."""
from visualize import save_matrices, save_dance, save_circles_dance
from multiprocessing import Pool
from PIL import Image
import numpy as np
import itertools
import argparse
import librosa
import madmom
import random
import scipy

random.seed(123)

# parse arguments
voice_parser = argparse.ArgumentParser(description='Process arguments.')
voice_parser.add_argument('-songpath', '--songpath', type=str, default='./audio_files/vocal.mp3',
                    help='Path to .mp3 song')
voice_parser.add_argument('-songname', '--songname', type=str, default='We will rock you',
                    help='Name of song')
voice_parser.add_argument('-steps', '--steps', type=int, default=100,
                    help='Number of equidistant LRS steps the agent should take')
voice_parser.add_argument('-type', '--type', type=str, default='action',
                    help='Type of dance -- state, action, stateplusaction')
voice_parser.add_argument('-baseline', '--baseline', type=str, default='none',
                    help='Generate baseline -- none, unsync_random, unsync_sequential, sync_sequential, sync_random')
voice_parser.add_argument('-visfolder', '--visfolder', type=str, default='./vis_num_steps_20/dancing_color_20',
                    help='path to folder containing agent visualizations')
voice_args = voice_parser.parse_args()

instrumental_parser = argparse.ArgumentParser(description='Process arguments.')
instrumental_parser.add_argument('-songpath', '--songpath', type=str, default='./audio_files/bakcgroundmusic.mp3',
                    help='Path to .mp3 song')
instrumental_parser.add_argument('-steps', '--steps', type=int, default=100,
                    help='Number of equidistant LRS steps the agent should take')
instrumental_parser.add_argument('-type', '--type', type=str, default='action',
                    help='Type of dance -- state, action, stateplusaction')
instrumental_parser.add_argument('-baseline', '--baseline', type=str, default='none',
                    help='Generate baseline -- none, unsync_random, unsync_sequential, sync_sequential, sync_random')
instrumental_args = instrumental_parser.parse_args()

# global variables
GRID_SIZE = 20
REWARD_INTERVAL = 5
ALL_ACTION_COMBS = list(set(itertools.permutations([-1 for _ in range(REWARD_INTERVAL)] + [1 for _ in range(REWARD_INTERVAL)] + [0 for _ in range(REWARD_INTERVAL)], REWARD_INTERVAL)))
START_POSITION = int(GRID_SIZE / 2)
ACTION_MAPPING = {-1: 'L', 1: 'R', 0: 'S'}    # L: left, R: right, S; stay
MUSIC_MODE = 'affinity'
MUSIC_METRIC = 'euclidean'
HOP_LENGTH = 512

# **************************************************************************************************************** #
# BASELINES


def unsync_random(num_steps):
    """Baseline 1 : unsynced - random."""
    states, actions = [], []
    curr = START_POSITION

    # get state and action sequences
    for i in range(num_steps):
        act = random.choice([-1, 0, 1])
        newcurr = curr + act
        if newcurr < 0:
            curr = 0
        elif newcurr == GRID_SIZE:
            curr = GRID_SIZE - 1
        else:
            curr = newcurr
        states.append(curr)
        actions.append(act)
    return [states, actions]


def unsync_sequential(num_steps):
    """Baseline 2 : unsynced - left2right."""
    states, actions = [], []
    curr = START_POSITION
    curr, prev = START_POSITION, START_POSITION - 1

    # get state and action sequences
    for i in range(num_steps):
        act = (curr - prev)
        newcurr = curr + act
        if newcurr < 0:
            prev = 0
            curr = 1
        elif newcurr == GRID_SIZE:
            prev = GRID_SIZE - 1
            curr = GRID_SIZE - 2
        else:
            prev = curr
            curr = newcurr
        states.append(curr)
        actions.append(act)
    return [states, actions]


def sync_sequential(num_steps, filename, duration):
    """Baseline 3 : synced - left2right."""
    # get beat information
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(filename)
    beat_times = np.around(proc(act) * num_steps / duration)

    states, actions = [], []
    curr = START_POSITION
    curr, prev = START_POSITION, START_POSITION - 1

    for i in range(num_steps):
        if i in beat_times:
            act = (curr - prev)
            newcurr = curr + act
            if newcurr < 0:
                prev = 0
                curr = 1
            elif newcurr == GRID_SIZE:
                prev = GRID_SIZE - 1
                curr = GRID_SIZE - 2
            else:
                prev = curr
                curr = newcurr
        else:
            act = 0
        states.append(curr)
        actions.append(act)
    return [states, actions]


def sync_random(num_steps, filename, duration):
    """Baseline 4 : synced - random."""
    # get beat information
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(filename)
    beat_times = np.around(proc(act) * num_steps / duration)

    states, actions = [], []
    curr = START_POSITION

    for i in range(num_steps):
        if i in beat_times:
            act = random.choice([-1, 0, 1])
            newcurr = curr + act
            if newcurr < 0:
                curr = 0
            elif newcurr == GRID_SIZE:
                curr = GRID_SIZE - 1
            else:
                curr = newcurr
        else:
            act = 0
        states.append(curr)
        actions.append(act)
    return [states, actions]

# **************************************************************************************************************** #
# DANCE MATRIX CREATION


def fill_dance_aff_matrix_diststate(states):
    """Fill state action affinity matrix - relative distance based states."""
    s = len(states)
    rowtile = np.tile(states, (s, 1))
    coltile = rowtile.T
    sa_aff = 1. - np.abs(rowtile-coltile) / (GRID_SIZE-1)
    # sa_aff = (sa_aff - np.min(sa_aff)) / (np.max(sa_aff) - np.min(sa_aff))    # normalize
    return sa_aff


def fill_dance_aff_matrix_action(actions):
    """Fill state action affinity matrix - action based."""
    s = len(actions)
    rowtile = np.tile(actions, (s, 1))
    coltile = rowtile.T
    sa_aff = np.maximum(1. - np.abs(rowtile-coltile), 0.)
    # sa_aff = (sa_aff - np.min(sa_aff)) / (np.max(sa_aff) - np.min(sa_aff))    # normalize
    return sa_aff


def fill_dance_aff_matrix_diststateplusaction(states, actions):
    """Fill state action affinity matrix - action based."""
    state_matrix = fill_dance_aff_matrix_diststate(states)
    action_matrix = fill_dance_aff_matrix_action(actions)
    sa_aff = (state_matrix + action_matrix) / 2.
    # sa_aff = (sa_aff - np.min(sa_aff)) / (np.max(sa_aff) - np.min(sa_aff))    # normalize
    return sa_aff


def get_dance_matrix(states, actions, dance_matrix_type, music_matrix_full):
    """Pass to appropriate dance matrix generation function based on dance_matrix_type."""
    if dance_matrix_type == 'state':
        dance_matrix = fill_dance_aff_matrix_diststate(states)
    elif dance_matrix_type == 'action':
        dance_matrix = fill_dance_aff_matrix_action(actions)
    elif dance_matrix_type == 'stateplusaction':
        dance_matrix = fill_dance_aff_matrix_diststateplusaction(states, actions)
    else:
        print("err")
    dance_matrix = np.array(Image.fromarray(np.uint8(dance_matrix * 255)).resize(music_matrix_full.shape, Image.NEAREST)) / 255.
    return dance_matrix

# **************************************************************************************************************** #
# MUSIC MATRIX COMPUTATION


def compute_music_matrix(y, sr, mode, metric):
    """Return music affinity matrix based on mode."""
    lifter = 0
    n_mfcc = 20
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, lifter=lifter, hop_length=HOP_LENGTH)
    R = librosa.segment.recurrence_matrix(mfcc, metric=metric, mode=mode, sym=True).T.astype(float)    # already normalized in 0-1
    np.fill_diagonal(R, 1)    # make diagonal entries 1
    return R

# **************************************************************************************************************** #
# REWARD COMPUTATION


def music_reward(music_matrix, dance_matrix, mtype):
    """Return the reward given music matrix and dance matrix."""
    # compute distance based on mtype
    if mtype == 'pearson':
        if np.array(music_matrix).std() == 0 or np.array(dance_matrix).std() == 0:
            reward = 0
        else:
            reward, p_val = scipy.stats.pearsonr(music_matrix.flatten(), dance_matrix.flatten())
    elif mtype == 'spearman':
        reward, p_val = scipy.stats.spearmanr(music_matrix.flatten(), dance_matrix.flatten())
    else:
        print("err")
    return reward


# **************************************************************************************************************** #
# BRUTE FORCE METHODS


def get_rsa_for_actionset(args):
    """Return reward, state, action set."""
    actionset, music_matrix, loc, num_actions, prev_states, prev_actions, dance_matrix_type = args
    curr_states = []
    curr_actions = []
    start_loc = loc
    for action in list(actionset):
        newpos = start_loc + action
        if newpos == 0 or newpos == GRID_SIZE:
            # hit wall
            break
        curr_states.append(newpos)
        curr_actions.append(action)
        start_loc = newpos

    # if not completed, ignore
    if len(curr_actions) != num_actions:
        return False, [], []

    # get dance up till now
    states = prev_states + curr_states
    actions = prev_actions + curr_actions

    # get dance matrix
    if dance_matrix_type == 'state':
        dance_matrix = fill_dance_aff_matrix_diststate(states)
    elif dance_matrix_type == 'action':
        dance_matrix = fill_dance_aff_matrix_action(actions)
    else:
        dance_matrix = fill_dance_aff_matrix_diststateplusaction(states, actions)
    dance_matrix = np.array(Image.fromarray(np.uint8(dance_matrix * 255)).resize(music_matrix.shape, Image.NEAREST)) / 255.
    # check how good dance up till now is by computing reward
    curr_reward = music_reward(music_matrix, dance_matrix, 'pearson')
    return curr_reward, states, actions


def getbest(loc, num_actions, prev_states, prev_actions, music_matrix_full, num_steps, dance_matrix_type):
    """Return best combination of size num_actions.
    Start from `loc` in grid of size `GRID_SIZE`.
    """
    scale = int(music_matrix_full.shape[0] * (len(prev_states)+num_actions) / num_steps)
    music_matrix = np.array([music_matrix_full[i][:scale] for i in range(scale)])
    # get best dance for this music matrix
    bestreward = 0
    #p = Pool()
    args = ((actionset, music_matrix, loc, num_actions, prev_states, prev_actions, dance_matrix_type) for actionset in ALL_ACTION_COMBS)
    res = map(get_rsa_for_actionset, args)
    # p.close()
    for curr_reward, states, actions in res:
        if curr_reward is not False and curr_reward > bestreward:
            bestreward = curr_reward
            beststates = states
            bestactions = actions
    return beststates, bestactions, bestreward

# **************************************************************************************************************** #
# MAIN


if __name__ == "__main__":

    # get args
    voice_songname = voice_args.songname
    voice_baseline = voice_args.baseline
    voice_dance_matrix_type = voice_args.type
    voice_num_steps = voice_args.steps
    voice_visfolder = voice_args.visfolder
    voice_songpath = voice_args.songpath

    instrumental_baseline = instrumental_args.baseline
    instrumental_dance_matrix_type = instrumental_args.type
    instrumental_songpath = instrumental_args.songpath

    # load song
    y1, sr1 = librosa.load(voice_songpath)    # default sampling rate 22050
    voice_duration = librosa.get_duration(y=y1, sr=sr1)

    y2, sr2 = librosa.load(instrumental_songpath)    # default sampling rate 22050
    instrumental_duration = librosa.get_duration(y=y2, sr=sr2)

    # get music matrix
    voice_music_matrix_full = compute_music_matrix(y1, sr1, MUSIC_MODE, MUSIC_METRIC)
    instrumental_music_matrix_full = compute_music_matrix(y2, sr2, MUSIC_MODE, MUSIC_METRIC)

    # main code for voice seperated part of song
    if voice_baseline in ['unsync_random', 'unsync_sequential', 'sync_sequential', 'sync_random']:
        # check baselines
        if voice_baseline == 'unsync_random':
            states, actions = unsync_random(num_steps=voice_num_steps)
        if voice_baseline == 'unsync_sequential':
            states, actions = unsync_sequential(num_steps=voice_num_steps)
        if voice_baseline == 'sync_sequential':
            states, actions = sync_sequential(num_steps=voice_num_steps, filename=voice_songpath, duration=voice_duration)
        else:
            states, actions = sync_random(num_steps=voice_num_steps, filename=voice_songpath, duration=voice_duration)

        # compute dance matrix
        dance_matrix = get_dance_matrix(states, actions, voice_dance_matrix_type, voice_music_matrix_full)

        # compute correlation
        reward = music_reward(voice_music_matrix_full, dance_matrix, 'pearson')
    else:
        # our approach
        # try out all combinations of `REWARD_INTERVAL` actions and compute reward
        voice_prev_states = []
        voice_prev_actions = []

        for i in range(voice_num_steps):
            # apply greedy algo to get dance matrix with best reward
            if len(voice_prev_actions) == 0:
                voice_prev_states, voice_prev_actions, voice_reward = getbest(loc=START_POSITION,
                                                            num_actions=REWARD_INTERVAL,
                                                            prev_states=voice_prev_states,
                                                            prev_actions=voice_prev_actions,
                                                            music_matrix_full=voice_music_matrix_full,
                                                            num_steps=voice_num_steps,
                                                            dance_matrix_type=voice_dance_matrix_type)
            elif not i % REWARD_INTERVAL:
                voice_prev_states, voice_prev_actions, voice_reward = getbest(loc=voice_prev_states[-1],
                                                            num_actions=REWARD_INTERVAL,
                                                            prev_states=voice_prev_states,
                                                            prev_actions=voice_prev_actions,
                                                            music_matrix_full=voice_music_matrix_full,
                                                            num_steps=voice_num_steps,
                                                            dance_matrix_type=voice_dance_matrix_type)
            elif voice_num_steps - len(voice_prev_actions) != 0 and voice_num_steps - len(voice_prev_actions) < REWARD_INTERVAL:
                voice_prev_states, voice_prev_actions, voice_reward = getbest(loc=voice_prev_states[-1],
                                                            num_actions=voice_num_steps-len(voice_prev_states),
                                                            prev_states=voice_prev_states,
                                                            prev_actions=voice_prev_actions,
                                                            music_matrix_full=voice_music_matrix_full,
                                                            num_steps=voice_num_steps,
                                                            dance_matrix_type=voice_dance_matrix_type)
            else:
                continue
    # main code for instrumental
    if instrumental_baseline in ['unsync_random', 'unsync_sequential', 'sync_sequential', 'sync_random']:
        # check baselines
        if instrumental_baseline == 'unsync_random':
            states, actions = unsync_random(num_steps=voice_num_steps)
        if instrumental_baseline == 'unsync_sequential':
            states, actions = unsync_sequential(num_steps=voice_num_steps)
        if instrumental_baseline == 'sync_sequential':
            states, actions = sync_sequential(num_steps=voice_num_steps, filename=instrumental_songpath, duration=duration)
        else:
            states, actions = sync_random(num_steps=voice_num_steps, filename=instrumental_songpath, duration=duration)

        # compute dance matrix
        dance_matrix = get_dance_matrix(states, actions, instrumental_dance_matrix_type, voice_music_matrix_full)

        # compute correlation
        reward = music_reward(instrumental_music_matrix_full, dance_matrix, 'pearson')
    else:
        # our approach
        # try out all combinations of `REWARD_INTERVAL` actions and compute reward
        instrumental_prev_states = []
        instrumental_prev_actions = []

        for i in range(voice_num_steps):
            # apply greedy algo to get dance matrix with best reward
            if len(instrumental_prev_actions) == 0:
                instrumental_prev_states, instrumental_prev_actions, instrumental_reward = getbest(loc=START_POSITION,
                                                            num_actions=REWARD_INTERVAL,
                                                            prev_states=instrumental_prev_states,
                                                            prev_actions=instrumental_prev_actions,
                                                            music_matrix_full=instrumental_music_matrix_full,
                                                            num_steps=voice_num_steps,
                                                            dance_matrix_type=instrumental_dance_matrix_type)
            elif not i % REWARD_INTERVAL:
                instrumental_prev_states, instrumental_prev_actions, instrumental_reward = getbest(loc=instrumental_prev_states[-1],
                                                            num_actions=REWARD_INTERVAL,
                                                            prev_states=instrumental_prev_states,
                                                            prev_actions=instrumental_prev_actions,
                                                            music_matrix_full=instrumental_music_matrix_full,
                                                            num_steps=voice_num_steps,
                                                            dance_matrix_type=instrumental_dance_matrix_type)
            elif voice_num_steps - len(instrumental_prev_actions) != 0 and voice_num_steps - len(instrumental_prev_actions) < REWARD_INTERVAL:
                instrumental_prev_states, instrumental_prev_actions, instrumental_reward = getbest(loc=instrumental_prev_states[-1],
                                                            num_actions=voice_num_steps-len(instrumental_prev_states),
                                                            prev_states=instrumental_prev_states,
                                                            prev_actions=instrumental_prev_actions,
                                                            music_matrix_full=instrumental_music_matrix_full,
                                                            num_steps=voice_num_steps,
                                                            dance_matrix_type=instrumental_dance_matrix_type)
            else:
                continue
        # get best dance matrix
        instrumental_dance_matrix = get_dance_matrix(instrumental_prev_states, instrumental_prev_actions, instrumental_dance_matrix_type, instrumental_music_matrix_full)
        voice_dance_matrix = get_dance_matrix(instrumental_prev_states, instrumental_prev_actions, voice_dance_matrix_type, instrumental_music_matrix_full)
        # assign states and actions correctly correctly
        instrumental_states = instrumental_prev_states
        instrumental_actions = instrumental_prev_actions

        voice_states = voice_prev_states
        voice_actions = voice_prev_actions

    # map actions correctly
    voice_actions = [ACTION_MAPPING[a] for a in voice_actions]
    instrumental_actions = [ACTION_MAPPING[a] for a in instrumental_actions]

    # print results
    print("Voice Correlation = ", voice_reward)
    print("Voice State sequence = ", voice_states)
    print("Voice Action sequence = ", voice_actions)

    print("Instrumental Correlation = ", instrumental_reward)
    print("Instrumental State sequence = ", instrumental_states)
    print("Instrumental Action sequence = ", instrumental_actions)

    # visualize results
     #save_matrices(music_matrix=voice_music_matrix_full, dance_matrix=dance_matrix, duration=duration)
    save_circles_dance(voice_states=voice_states, instrumental_states=instrumental_states, visfolder=voice_visfolder, songname=voice_songname, duration=voice_duration, num_steps=voice_num_steps)

    print("WeWillRockYou", " :: DONE!")