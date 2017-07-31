#!/usr/bin/env python
"""
example for thor environment
"""
import json
import robosims
import cv2
import numpy as np
import time

env = robosims.controller.ChallengeController(
        unity_path='projects/thor-201706291201-Linux64',
        x_display="0.0")
env.start()
actions = ['MoveAhead',
           'MoveRight', 
           'MoveLeft', 
           'MoveBack', 
           'LookUp', 
           'LookDown',
           'RotateRight',
           'RotateLeft']
with open("thor-challenge-targets/targets-val.json") as f:
  t = json.loads(f.read())
  total_start = time.time()
  total_target = len(t)
  found_target = 0
  for target in t:
    num_steps = 0
    env.initialize_target(target)
    start = time.time()
    print('Need to find {}'.format(target['targetImage']))
    while not env.target_found() and num_steps < 500:
      action = np.random.choice(actions)
      event = env.step(action=dict(action=action))
      num_steps += 1
      image = event.frame
    if env.target_found():
      print('Have found target in {} steps'.format(num_steps))
      found_target += 1
    else:
      print('Not found target after {} steps and give up'.format(num_steps))
    end = time.time()
    print('{} steps per second'.format(num_steps/(end-start)))
    print('*****init new target*****')
  total_end = time.time()
  print('total execution time {} seconds'.format(total_end - total_start))
  print('found {} in {}'.format(found_target, total_target))
