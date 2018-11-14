/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_input_layer.h"
#include "pt_layer_data.h"

namespace pt
{


bool InputLayer::apply(LayerData& layerData) const
{
    layerData.out = layerData.in;
    return true;
}
    
std::unique_ptr<InputLayer> InputLayer::create(std::istream& stream)
{

    return std::unique_ptr<InputLayer>(new InputLayer());
}


}
