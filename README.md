# Forked from @purvaten



# (Not official) Feel The Music: Automatically Generating A Dance For An Input Song

Full text available at: https://arxiv.org/abs/2006.11905

## Requirements
Create a new Python 3.7 virtual environment. Install the requirements using
```pip install -r requirements.txt```

## Steps for Generating Dances
0. ```git clone https://github.com/purvaten/feel-the-music.git```

1. ```cd feel-the-music```

2. Generate dance (example below)
```
python generate_dance.py \
--songpath './audio_files/flutesong.mp3' \
--songname 'flutesong' \
--steps 100 \
--type "action" \
--visfolder './vis_num_steps_20/dancing_person_20'
```

A folder named `plots` will be created in the current directory containing frames of the dance and the final combined output as `<songname>.mp4`. The music and dance matrices will be saved as `music.png` and `dance.png` in the current directory.

**NOTE** : `<visfolder>` should contain `GRID_SIZE` number of images of the agent, smoothly transitioning into each other with the files numbered as `1.png`, `2.png`, ... `<GRID_SIZE>.png`. In our experiments `GRID_SIZE=20`.

## Results
| Song | Type | Number of Steps | Agent | Video |
| --- | --- | --- | --- | --- |
| flutesong | action | 100 | Stick figure | <a href="https://s3.amazonaws.com/dancing-agents/results/flutesong/dance6.mp4"><img alt="Qries" src="https://user-images.githubusercontent.com/13128829/85195508-1335e780-b2a1-11ea-8a57-e64c776b6a56.jpeg" width="200"></a> |
| Oe oe oe oa | action | 50 | Stretchy leaves | <a href="https://s3.amazonaws.com/dancing-agents/results/oeoe/dance3_50.mp4"><img alt="Qries" src="https://user-images.githubusercontent.com/13128829/85196165-33b47080-b2a6-11ea-9a37-4da3d820b39d.png" width="200"></a> |
| It's the time to disco (karaoke) | action | 100 | Floating leaves | <a href="https://s3.amazonaws.com/dancing-agents/results/itsthetimetodiscono/dance1.mp4"><img alt="Qries" src="https://user-images.githubusercontent.com/13128829/85195869-f5b64d00-b2a3-11ea-93ea-b8c86bc55952.png" width="200"></a> |

For more examples, check https://sites.google.com/view/dancing-agents.
