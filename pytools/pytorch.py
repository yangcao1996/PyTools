import torch
import numpy as np

def print_model_param(model):
  if type(model) == dict:
    model_dict = model
  else:
    model_dict = model.state_dict()
  for k, v in model_dict.items():
    # print(v.device())
    pnumpy = v.cpu().numpy()
    print("%s, mean=%f, std=%f" % (k, pnumpy.mean(), pnumpy.std()))
