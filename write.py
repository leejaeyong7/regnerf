# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation script for RegNeRF."""
import functools
from os import path
import time

from absl import app
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
from internal import configs, datasets, math, models, utils, vis  # pylint: disable=g-multiple-import
import jax
from jax import random
import numpy as np
from skimage.metrics import structural_similarity
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
import shutil

CENSUS_EPSILON = 1 / 256  # Guard against ground-truth quantization.

configs.define_common_flags()
jax.config.parse_flags_with_absl()


def main(unused_argv):

  tf.config.experimental.set_visible_devices([], 'GPU')
  tf.config.experimental.set_visible_devices([], 'TPU')

  config = configs.load_config(save_config=False)

  dataset = datasets.load_dataset('all', config.data_dir, config)

  model, init_variables = models.construct_mipnerf(
      random.PRNGKey(20200823),
      dataset.peek()['rays'],
      config)
  optimizer = flax.optim.Adam(config.lr_init).create(init_variables)
  state = utils.TrainState(optimizer=optimizer)
  del optimizer, init_variables

  # Rendering is forced to be deterministic even if training was randomized, as
  # this eliminates 'speckle' artifacts.
  def render_eval_fn(variables, _, rays):
    return jax.lax.all_gather(
        model.apply(
            variables,
            None,  # Deterministic.
            rays,
            resample_padding=config.resample_padding_final,
            compute_extras=True), axis_name='batch')

  # pmap over only the data input.
  render_eval_pfn = jax.pmap(
      render_eval_fn,
      in_axes=(None, None, 0),
      donate_argnums=2,
      axis_name='batch',
  )

  def ssim_fn(x, y):
    return structural_similarity(x, y, multichannel=True)

  census_fn = jax.jit(
      functools.partial(math.compute_census_err, epsilon=CENSUS_EPSILON))

  print('WARNING: LPIPS calculation not supported. NaN values used instead.')
  if config.eval_disable_lpips:
    lpips_fn = lambda x, y: np.nan
  else:
    lpips_fn = lambda x, y: np.nan

  last_step = 0
  out_dir = path.join(config.checkpoint_dir, 'results')
  path_fn = lambda x: path.join(out_dir, x)

  output_path = Path(out_dir)
  output_path.mkdir(exist_ok=True, parents=True)
  (output_path / 'images').mkdir(exist_ok=True, parents=True)
  (output_path / 'depths').mkdir(exist_ok=True, parents=True)

  data_path = Path(config.data_dir)
  shutil.copy(data_path / 'transforms.json', output_path / 'transforms.json')

  while True:
    # Fix for loading pre-trained models.
    try:
      state = checkpoints.restore_checkpoint(config.checkpoint_dir, state)
    except:  # pylint: disable=bare-except
      print('Using pre-trained model.')
      state_dict = checkpoints.restore_checkpoint(config.checkpoint_dir, None)
      for i in [9, 17]:
        del state_dict['optimizer']['target']['params']['MLP_0'][f'Dense_{i}']
      state_dict['optimizer']['target']['params']['MLP_0'][
          'Dense_9'] = state_dict['optimizer']['target']['params']['MLP_0'][
              'Dense_18']
      state_dict['optimizer']['target']['params']['MLP_0'][
          'Dense_10'] = state_dict['optimizer']['target']['params']['MLP_0'][
              'Dense_19']
      state_dict['optimizer']['target']['params']['MLP_0'][
          'Dense_11'] = state_dict['optimizer']['target']['params']['MLP_0'][
              'Dense_20']
      del state_dict['optimizerd']
      state = flax.serialization.from_state_dict(state, state_dict)

    step = int(state.optimizer.state.step)
    if step <= last_step:
      print(f'Checkpoint step {step} <= last step {last_step}, sleeping.')
      time.sleep(10)
      continue
    print(f'Evaluating checkpoint at step {step}.')
    if config.eval_save_output and (not utils.isdir(out_dir)):
      utils.makedirs(out_dir)

    key = random.PRNGKey(0 if config.deterministic_showcase else step)
    perm = random.permutation(key, dataset.size)
    showcase_indices = np.sort(perm[:config.num_showcase_images])

    metrics = []
    showcases = []
    for idx in tqdm(range(dataset.size), total=dataset.size, dynamic_ncols=True):
      batch = next(dataset)
      rendering = models.render_image(
          functools.partial(render_eval_pfn, state.optimizer.target),
          batch['rays'],
          None,
          config)

      utils.save_img_u8(rendering['rgb'], output_path / 'images' / f'{idx:06d}.rgb.png')
      utils.save_img_f32(rendering['distance_mean'], output_path / 'depths' / f'{idx:06d}.depth.tiff')
      utils.save_img_f32(rendering['distance_median'], output_path / 'depths' / f'{idx:06d}.depthm.tiff')
    break


if __name__ == '__main__':
  app.run(main)
