/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_global_max_pooling_2d_layer.h"

#include <array>
#include "pt_parser.h"
#include "pt_layer_data.h"
#include "pt_max.h"

namespace pt
{

namespace
{
    void maxImpl(LayerData& layerData)
    {
        const Tensor& in = layerData.in;
        Tensor& out = layerData.out;

        const auto& iw = in.getDims();
        const auto& ow = out.getDims();

        auto inData = in.getData().data();
        auto outData = const_cast<Tensor::Type*>(out.getData().data());

        size_t its = ow[2];

        for (std::size_t z = 0; z < its; z++ )
        {
            Tensor::Type val = std::numeric_limits<Tensor::Type>::lowest();
            for (std::size_t x = 0; x < iw[0]; ++x)
            {
                for (std::size_t y = 0; y < iw[1]; ++y)
                {
                    val = std::max(val, in(x, y, z));
                }
            }
            out(0, 0, z) = val;
        }
    }
}

std::unique_ptr<GlobalMaxPooling2DLayer> GlobalMaxPooling2DLayer::create(std::istream& stream)
{

    return std::unique_ptr<GlobalMaxPooling2DLayer>(new GlobalMaxPooling2DLayer());
}

bool GlobalMaxPooling2DLayer::apply(LayerData& layerData) const
{
    const Tensor& in = layerData.in;
    const auto& iw = in.getDims();

    if(iw.size() != 3)
    {
        PT_LOG_ERROR << "Input tensor dims count must be 3" <<
                            " (input dims: " << VectorPrinter<std::size_t>{ iw } << ")" << std::endl;
        return false;
    }

    Tensor& out = layerData.out;
    out.resize(1, 1, iw[2]);
    out.fill(-std::numeric_limits<Tensor::Type>::infinity());

    maxImpl(layerData);
    out.flatten();

    return true;
}

}
