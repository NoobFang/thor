#!/usr/bin/env python
# -*- coding: utf-8 -*-
# thor player for interaction with Thor environment

import numpy as numpy
import time
import os
import cv2
from collections import deque
import six
from six.moves import range
import json
import robosims

from tensorpack.utils import (get_rng, logger, execute_only_once)
from tensorpack.utils.fs import get_dataset_path
from tensorpack.utils.stats import StatCounter
from tensorpack.RL.envbase import RLEnvironment, DiscreteActionSpace

__all__ = ['ThorPlayer']

ACTIONS = ['MoveAhead',
           'MoveRight', 
           'MoveLeft', 
           'MoveBack', 
           'LookUp', 
           'LookDown',
           'RotateRight',
           'RotateLeft']

HEIGHT = 300
WIDTH = 300

class ThorPlayer(RLEnvironment):
  """
  a wrapper for Thor environment.
  """
  def __init__(self, exe_path, json_path, actions=ACTIONS, height=HEIGHT, width=WIDTH, gray=False, record=False):
    super(ThorPlayer, self).__init__()
    assert os.path.isfile(exe_path), 'wrong path of executable binary for Thor'
    assert os.path.isfile(json_path), 'wrong path of target json file'
    self.height = height
    self.width = width
    self.gray = gray
    self.record = record
    # set Thor controller
    self.env = robosims.controller.ChallengeController(
                unity_path=exe_path,
                height=self.height,
                width=self.width,
                record_actions=self.record)
    
    # read targets from the json file
    with open(json_path) as f:
      self.targets = json.loads(f.read())
    self.num_targets = len(self.targets)
    
    self.rng = get_rng(self)
    self.actions = actions
    self.current_episode_score = StatCounter()
    self.env.start()
    self.restart_episode()

  def current_state(self):
    # image of current state, numpy array of (h, w, 3) in RGB order
    img = self.env.last_event.frame
    success = self.env.last_event.metadata['lastActionSuccess']
    found = self.env.target_found()
    if self.gray:
      img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)]
    
    return img, success, found

  def get_action_space(self):
    return DiscreteActionSpace(len(self.actions))

  def next_target(self):
    idx = self.rng.choice(range(self.num_targets))
    return self.targets[idx]

  def restart_episode(self):
    """
    reset the episode counter and
    initialize the env by a random selected target
    """
    self.current_episode_score.reset()
    target = self.next_target()
    self.env.initialize_target(target)

  def action(self, act):
    """
    Perform an action.
    Will automatically start a new episode if isOver
    """
    r = 0.0
    isOver = False
    event = self.env.step(action=dict(action=self.actions[act]))
    if not event.metadata['lastActionSuccess']:
      r -= 0.01
    if self.env.target_found():
      r += 100.0
      isOver = True
    self.current_episode_score.feed(r)
    if isOver:
      self.restart_episode()
    return (r, isOver)

