# encoding=utf8
# 在cuda pytorch0.4.0环境下训练的模型转换到CPU机器pytorch0.3.1

from torch import _utils
from torch._utils import _rebuild_tensor
from torch.serialization import load, save

def _rebuild_tensor_v2(storage, storage_offset, size, stride, required_grad, backward_hooks):
    tensor = _rebuild_tensor(storage, storage_offset, size, stride)
    tensor.requires_grad = required_grad
    tensor._backward_hooks = backward_hooks
    return tensor

_utils._rebuild_tensor_v2 = _rebuild_tensor_v2

modelpath = 'seq2seq/model.th'
modelpath_new = modelpath + '.0.3.1'

m = load(modelpath, map_location='cpu')
save(m, modelpath_new)

