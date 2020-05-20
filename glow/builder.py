import re, os
import copy
from .config import JsonConfig
from .models import Glow
from torch.utils.data import DataLoader
import numpy as np
from mindspore.train.serialization import load_checkpoint, save_checkpoint, load_param_into_net
import torch
from mindspore import Tensor
import sys

def save_glow_fwd_model(fwd_net, fwd_net_ckpt_file):
    param_list = []
    param_dict = {}
    for _, param in fwd_net.parameters_and_names():
        param_dict[param.name] = param
    for (key, value) in param_dict.items():
        each_param = {}
        each_param["name"] = key
        each_param["data"] = value.data
        param_list.append(each_param)
    save_checkpoint(param_list, fwd_net_ckpt_file)
    return

# load ms checkpoint
def load_ms_pretrained_model(graph, hparams):
    assert os.path.exists(hparams.Infer.ms_pre_trained), (
        "Failed to find ms_ckpt_file `{}`".format(hparams.Infer.ms_pre_trained))
    ms_ckpt = load_checkpoint(hparams.Infer.ms_pre_trained)
    load_param_into_net(graph, ms_ckpt)
    return graph

# load pytorch checkpoint
def load_pt_pretrained_model(graph, hparams):
    print("Both of the mindspore checkpoint file and pytorch checkpoint file must be provided:")
    assert os.path.exists(hparams.Infer.ms_pre_trained), (
        "Failed to find ms_ckpt_file `{}`".format(hparams.Infer.ms_pre_trained))
    assert os.path.exists(hparams.Infer.pt_pre_trained), (
        "Failed to find pt_ckpt_file `{}`".format(hparams.Infer.pt_pre_trained))
    ms_ckpt = load_checkpoint(hparams.Infer.ms_pre_trained)
    pt_ckpt = torch.load(hparams.Infer.pt_pre_trained,map_location = torch.device("cpu"))
    for pt_k, pt_v in pt_ckpt.items():
        if pt_k == "graph":
            pt_ckpt = pt_v
            break
    new_params_list=[]
    for ms_k, ms_v in ms_ckpt.items():
        if ms_k.find("accumulation") >= 0:
            continue
        t_pt_k = ms_k.replace("layer_", "layers.")
        t_pt_k = t_pt_k.replace("conv2d.", "")
        t_pt_k = t_pt_k.replace("conv1", "0")
        t_pt_k = t_pt_k.replace("conv2", "2")
        t_pt_k = t_pt_k.replace("conv3", "4")
        t_pt_k = t_pt_k.replace("conv.", "", 1)
        t_pt_k = t_pt_k.replace("2dzeros", "conv", 1)
        t_pt_k = t_pt_k.replace("shuffled_indices", "indices_tensor")
        t_pt_k = t_pt_k.replace("inversed_indices", "indices_inverse_tensor")
        for pt_k, pt_v in pt_ckpt.items():
            if t_pt_k == pt_k and isinstance(ms_v.data, Tensor):
                param_dict = {}
                param_dict['name'] = ms_k
                if ms_v.data.shape() == pt_v.shape:
                    if ms_k.find("shuffle") >=0:
                        param_dict['data'] = Tensor(pt_v.numpy().astype(np.int32))
                    else:
                        param_dict['data'] = Tensor(pt_v.numpy().astype(np.float32))
                else:
                    if (ms_k.find("conv") >= 0) or (ms_k.find("learn_top") >= 0):
                        if ms_k.find("logs") >= 0:
                            new_pt_v = np.expand_dims(pt_v.numpy().astype(np.float32), axis=0)
                        elif ms_k.find("bias") >= 0:
                            new_pt_v = np.reshape(pt_v.numpy().astype(np.float32),[1,pt_v.shape[0],1,1])
                        else:
                            print("Error: unresovled error")
                    elif (ms_k.find("project_ycond") >= 0) or (ms_k.find("project_class") >= 0):
                            new_pt_v = np.expand_dims(pt_v.numpy().astype(np.float32), axis=0)
                            new_pt_v = np.expand_dims(pt_v.numpy().astype(np.float32), axis=0)
                    else:
                        print("Error: shape of %s does not match with pytorch", ms_k)
                    if ms_v.data.shape() == new_pt_v.shape:
                        param_dict['data'] = Tensor(new_pt_v)
                    else:
                        print(" Load pytorch model ERROR -- begin " * 5)
                        print(ms_k)
                        print("pt_shape: ", pt_v.shape)
                        print("new_pt_shape: ", new_pt_v.shape)
                        print("ms_shape: ", ms_v.data.shape())
                        print(" Load pytorch model ERROR -- end " * 5)
                new_params_list.append(param_dict)
                break
    if os.path.exists("load_from_pt.ckpt"):
        os.remove("load_from_pt.ckpt")
    save_checkpoint(new_params_list, "load_from_pt.ckpt")
    ms_ckpt = load_checkpoint("load_from_pt.ckpt")
    load_param_into_net(graph, ms_ckpt)
    return graph

def create_train_graph(hparams, dataset):
    hparams =  JsonConfig(hparams)
    if hparams.Glow.reverse is True:
        train_graph = Glow(hparams)
    else:
        init_graph_ckpt_file_name = "checkpoints/init_graph_L_" + str(hparams.Glow.L) + "_K_" + str(hparams.Glow.K) + ".ckpt"
        if os.path.exists(init_graph_ckpt_file_name):
            print("L= " + str(hparams.Glow.L) + " K= " + str(hparams.Glow.K) + " has initilized!")
        else:
            # init logs, bias, shuffle_indices
            init_graph = Glow(hparams, init=True)
            dataloader = DataLoader(dataset, batch_size=hparams.Train.batch_size, shuffle=False, drop_last=True)
            for data in dataloader:
                x = Tensor(data["image"].numpy() + np.random.normal(0., 1.0/256, size=(hparams.Train.batch_size, 3, 64, 64)).astype(np.float32))
                y_onehot = Tensor(data["attr"].numpy().astype(np.float32))
                break
            z,_,_,_ = init_graph(x, y_onehot)
            # save init_graph ckpt
            if not os.path.exists("checkpoints/"):
                os.makedirs("checkpoints/")
            save_glow_fwd_model(init_graph, init_graph_ckpt_file_name)
            # restart the program, to free the device memory occupied by init_graph
            sys.exit(123)
        # create train net
        train_graph = Glow(hparams)
        # load initial parameters
        ms_ckpt = load_checkpoint(init_graph_ckpt_file_name)
        load_param_into_net(train_graph, ms_ckpt)
    return train_graph

def build(hparams, dataset, is_training):
    if isinstance(hparams, str):
        hparams = JsonConfig(hparams)
    # get graph and criterions from build function
    graph, optim, lrschedule = None, None, None  # init with None
    devices = "cpu", None
    graph = create_train_graph(hparams, dataset)

    if hparams.Infer.load_ms_checkpoint is True:
        graph = load_ms_pretrained_model(graph, hparams)
    if hparams.Infer.load_pt_checkpoint is True:
        graph = load_pt_pretrained_model(graph, hparams)

    if hparams.Glow.reverse is True:
        graph_decoder = Glow(hparams, reverse=True)
        if hparams.Infer.load_ms_checkpoint is True:
            graph_decoder = load_ms_pretrained_model(graph_decoder, hparams)
        if hparams.Infer.load_pt_checkpoint is True:
            graph_decoder = load_pt_pretrained_model(graph_decoder, hparams)
        if hparams.Infer.load_ms_checkpoint or hparams.Infer.load_pt_checkpoint:
            for layer in reversed(graph_decoder.flow.layers):
                graph_decoder.flow.reversed_layer.append(layer)
            for i in range(len(graph_decoder.flow.reversed_layer)):
                exec("graph_decoder.flow.reversed_layer_" + str(i) + "= graph_decoder.flow.reversed_layer[" + str(i) + "]")
    else:
        graph_decoder = None
    return {
        "graph": graph,
        "graph_decoder": graph_decoder,
        "optim": optim,
        "lrschedule": lrschedule,
        "devices": devices
    }