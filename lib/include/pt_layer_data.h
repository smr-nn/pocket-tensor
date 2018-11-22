﻿/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_LAYER_DATA_H
#define PT_LAYER_DATA_H

#include "pt_tensor.h"

namespace pt
{

class Config;

struct LayerData
{
    Tensor in;
    Tensor& out;
    const Config& config;
};

}

#endif
