﻿/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_max_pooling_2d_layer.h"

#include <array>
#include "pt_parser.h"
#include "pt_layer_data.h"
#include "pt_max.h"

namespace pt
{

namespace
{
    template<class MaxType>
    void maxImpl(int poolSizeY, int poolSizeX, LayerData& layerData)
    {
        const Tensor& in = layerData.in;
        Tensor& out = layerData.out;

        const auto& iw = in.getDims();
        const auto& ow = out.getDims();
        auto inIncY2 = int(iw[2] * iw[1]);
        auto inIncY = inIncY2 * poolSizeY;
        auto inIncX2 = int(iw[2]);
        auto inIncX = inIncX2 * poolSizeX;
        auto outInc2 = int(iw[2] * ow[1]);
        auto outInc = outInc2 * int(ow[0]);

        auto inData = in.getData().data();
        auto outData = const_cast<Tensor::Type*>(out.getData().data());
        MaxType max;

        int its = outInc / outInc2;
        int taskEnd = its;

        for(auto outIt = outData, outEnd = outData + (taskEnd * outInc2);
            outIt != outEnd; outIt += outInc2)
        {
            auto inIt = inData;
            inData += inIncY;

            for(auto outIt2 = outIt, outEnd2 = outIt + outInc2; outIt2 != outEnd2; outIt2 += inIncX2)
            {
                for(auto inIt2 = inIt, inEnd2 = inIt + inIncY; inIt2 != inEnd2; inIt2 += inIncY2)
                {
                    for(auto inIt3 = inIt2, inEnd3 = inIt2 + inIncX; inIt3 != inEnd3; inIt3 += inIncX2)
                    {
                        max(&*inIt3, &*outIt2, inIncX2);
                    }
                }

                inIt += inIncX;
            }
        }
    }
}

std::unique_ptr<MaxPooling2DLayer> MaxPooling2DLayer::create(std::istream& stream)
{
    unsigned int poolSizeY = 0;

    if(! Parser::parse(stream, poolSizeY))
    {
        PT_LOG_ERROR << "Pool size Y parse failed" << std::endl;
        return std::unique_ptr<MaxPooling2DLayer>();
    }

    unsigned int poolSizeX = 0;

    if(! Parser::parse(stream, poolSizeX))
    {
        PT_LOG_ERROR << "Pool size X parse failed" << std::endl;
        return std::unique_ptr<MaxPooling2DLayer>();
    }

    return std::unique_ptr<MaxPooling2DLayer>(new MaxPooling2DLayer(int(poolSizeY), int(poolSizeX)));
}

bool MaxPooling2DLayer::apply(LayerData& layerData) const
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
    out.resize(iw[0] / std::size_t(_poolSizeY), iw[1] / std::size_t(_poolSizeX), iw[2]);
    out.fill(-std::numeric_limits<Tensor::Type>::infinity());

    auto tensorSize = int(iw[2]);

    if(PT_LOOP_UNROLLING_ENABLE && tensorSize && tensorSize % (Tensor::VectorSize * 2) == 0)
    {
        maxImpl<VectorMax>(_poolSizeY, _poolSizeX, layerData);
    }
    else if(tensorSize && tensorSize % Tensor::VectorSize == 0)
    {
        maxImpl<VectorMax>(_poolSizeY, _poolSizeX, layerData);
    }
    else
    {
        maxImpl<ScalarMax>(_poolSizeY, _poolSizeX, layerData);
    }

    return true;
}

}
