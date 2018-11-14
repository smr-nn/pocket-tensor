/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_INPUT_LAYER_H
#define PT_INPUT_LAYER_H

#include "pt_layer.h"
#include "pt_tensor.h"

#include <string>

namespace pt
{

class InputLayer : public Layer
{

public:
    static std::unique_ptr<InputLayer> create(std::istream& stream);

    bool apply(LayerData& layerData) const final;
    
    using DimsVector = std::vector<std::size_t>;
    // protected:
    //    
    //    std::string _name;
    //    DimsVector _dims;

    //InputLayer(const std::string& input_name, const DimsVector& input_dims)
    //    :   _name(input_name), _dims(input_dims)
    //{
    //}
    InputLayer(){}
};

}

#endif
