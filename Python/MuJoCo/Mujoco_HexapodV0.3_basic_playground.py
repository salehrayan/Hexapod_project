import numpy as np
import math

import mujoco
import mujoco.viewer
from mujoco import mjx

import jax
from jax import numpy as jp

import brax
from brax.envs.base import PipelineEnv, Env, State
from brax.io import mjcf, model

from moviepy.editor import ImageSequenceClip



with open(r'E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\final_noFrictionLoss_noCoxaCon_explicitConPair.xml', 'r') as f:
  xml = f.read()

# xml = """
# <mujoco>
#   <worldbody>
#     <light name="top" pos="0 0 1"/>
#     <body name="box_and_sphere" euler="0 0 -30">
#       <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
#       <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
#       <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
#     </body>
#   </worldbody>
# </mujoco>
# """

mj_model = mujoco.MjModel.from_xml_string(xml)
mj_data = mujoco.MjData(mj_model)
mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
mj_model.opt.iterations = 6
mj_model.opt.ls_iterations = 6

renderer = mujoco.Renderer(mj_model, height=480, width=640)
# mujoco.mj_resetData(mj_model, mj_data)

# jit_step = jax.jit(mjx.step)

# mjx_model = mjx.put_model(mj_model)
# mjx_data = mjx.put_data(mj_model, mj_data)


# print(mjx_data.time)
# mjx_data = jit_step(mjx_model, mjx_data)
#
# scene_option = mujoco.MjvOption()
# scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
#
# duration = 3
# fps = 60
#
# jit_step = jax.jit(mjx.step)
#
# frames = []
# mujoco.mj_resetData(mj_model, mj_data)
# mjx_data = jit_step(mjx_model, mjx_data)
# while mjx_data.time < duration:
#   mjx_data = jit_step(mjx_model, mjx_data)
#   if len(frames) < mjx_data.time * fps:
#     mj_data = mjx.get_data(mj_model, mjx_data)
#     renderer.update_scene(mj_data, scene_option=scene_option)
#     pixels = renderer.render()
#     frames.append(pixels)
# #
# clip = ImageSequenceClip(frames, fps=fps)
# clip.write_videofile('test.mp4', fps=fps)


