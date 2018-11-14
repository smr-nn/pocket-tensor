﻿/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_embedding_layer.h"

#include "pt_layer_data.h"
#include "pt_logger.h"

namespace pt
{

std::unique_ptr<EmbeddingLayer> EmbeddingLayer::create(std::istream& stream)
{
    auto weights = Tensor::create(2, stream);

    if(! weights)
    {
        PT_LOG_ERROR << "Weights tensor parse failed" << std::endl;
        return std::unique_ptr<EmbeddingLayer>();
    }

    return std::unique_ptr<EmbeddingLayer>(new EmbeddingLayer(std::move(*weights)));
}

bool EmbeddingLayer::apply(LayerData& layerData) const
{
    const Tensor& in = layerData.in;
    const auto& iw = in.getDims();

    Tensor& out = layerData.out;
    out.resize(iw[0], iw[1], _weights.getDims()[1]);

    auto outIt = out.begin();
    auto wBegin = _weights.begin();
    auto inc = _weights.getDims()[1];

    for(auto inIt = in.begin(), inEnd = in.begin() + long(iw[0] * iw[1]); inIt != inEnd; ++inIt)
    {
        auto wIt = wBegin + int(*inIt * inc);
        std::memcpy(&*outIt, &*wIt, inc * sizeof(Tensor::Type));
        outIt += long(inc);
    }

    return true;
}

EmbeddingLayer::EmbeddingLayer(Tensor&& weights) noexcept :
    _weights(std::move(weights))
{
}

}
