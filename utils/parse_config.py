import torch
import numpy as np


def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r', encoding='UTF-8')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0  # if the conv has batch ,the value will be covered..
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


def load_resnet_weigth(model, weigths_path):
    dict = torch.load(weigths_path)
    dict.pop("fc.weight")
    dict.pop("fc.bias")
    # dict = param.__reversed__()
    cache_weight = []
    cache_bias = []
    cache_running_mean = []
    cache_running_var = []
    with torch.no_grad():
        for module_i,(module_def,module) in enumerate(zip(model.module_defs, model.module_list)):
            if module_i == 6:
                break
            if module_def['type'] == "convolutional":
                conv_layer = module[0]
                conv_layer.weight.data.copy_(dict.popitem(last=False)[1])
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.weight.data.copy_(dict.popitem(last=False)[1])
                    bn_layer.bias.data.copy_(dict.popitem(last=False)[1])
                    bn_layer.running_mean.data.copy_(dict.popitem(last=False)[1])
                    bn_layer.running_var.data.copy_(dict.popitem(last=False)[1])
                else:
                    if module_i == 0:
                        continue
                    conv_layer.bias.data.copy_(dict.popitem(last=False))
            elif module_def['type'] == 'bottleneck':
                block = int(module_def["block"])
                for i in range(block):
                    little_module = module[i]
                    # conv one
                    bottle = little_module.bottleneck
                    list = [3,0,6,4,9,7]
                    for j in range(3):
                        index= j*2
                        print(str(module_i) + " :"+str(i) + "_"+str(j))
                        if j==0:
                            cache_weight.append((dict.popitem(last=False)[1]).numpy().tolist())
                            cache_bias.append((dict.popitem(last=False)[1]).numpy().tolist())
                            cache_running_mean.append((dict.popitem(last=False)[1]).numpy().tolist())
                            cache_running_var.append((dict.popitem(last=False)[1]).numpy().tolist())
                            if cache_weight.__len__() == 2:
                                bottle[list[index + 1]].weight.data.copy_(torch.from_numpy(np.mean(np.array(cache_weight),axis=0)))
                                bottle[list[index + 1]].bias.data.copy_(torch.from_numpy(np.mean(np.array(cache_bias),axis=0)))
                                bottle[list[index + 1]].running_mean.data.copy_(torch.from_numpy(np.mean(np.array(cache_running_mean),axis=0)))
                                bottle[list[index + 1]].running_var.data.copy_(torch.from_numpy(np.mean(np.array(cache_running_var),axis=0)))
                            else:
                                bottle[list[index + 1]].weight.data.copy_(
                                    torch.from_numpy(np.array(cache_weight)).squeeze(0))
                                bottle[list[index + 1]].bias.data.copy_(
                                    torch.from_numpy(np.array(cache_bias).reshape(1,-1)).squeeze(0))
                                bottle[list[index + 1]].running_mean.data.copy_(
                                    torch.from_numpy(np.array(cache_running_mean).reshape(1,-1)).squeeze(0))
                                bottle[list[index + 1]].running_var.data.copy_(
                                    torch.from_numpy(np.array(cache_running_var).reshape(1,-1)).squeeze(0))
                            cache_weight = []
                            cache_bias=[]
                            cache_running_var=[]
                            cache_running_mean=[]
                        else:
                            bottle[list[index + 1]].weight.data.copy_(dict.popitem(last=False)[1])
                            bottle[list[index + 1]].bias.data.copy_(dict.popitem(last=False)[1])
                            bottle[list[index + 1]].running_mean.data.copy_(dict.popitem(last=False)[1])
                            bottle[list[index + 1]].running_var.data.copy_(dict.popitem(last=False)[1])
                        k,m = dict.popitem(last=False)
                        print(k)
                        bottle[list[index]].weight.copy_(m)
                    if i == 0:
                        cache_weight.append((dict.popitem(last=False)[1]).numpy().tolist())
                        cache_bias.append((dict.popitem(last=False)[1]).numpy().tolist())
                        cache_bias.append((dict.popitem(last=False)[1]).numpy().tolist())
                        cache_bias.append((dict.popitem(last=False)[1]).numpy().tolist())
                        downsample = little_module.downsample
                        downsample[0].weight.data.copy_(dict.popitem(last=False)[1])
            elif module_def['type'] =='channel_selection':
                module[0].weight.data.copy_(dict.popitem(last=False)[1])
                module[0].bias.data.copy_(dict.popitem(last=False)[1])
                module[0].running_mean.data.copy_(dict.popitem(last=False)[1])
                module[0].running_var.data.copy_(dict.popitem(last=False)[1])