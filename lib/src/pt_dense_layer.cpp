﻿/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_dense_layer.h"

#include <array>
#include "pt_layer_data.h"
#include "pt_multiply_add.h"
#include "pt_logger.h"

namespace pt
{

namespace
{
    template<class MultiplyAddType>
    void multiplyAddImpl(const Tensor& weights, LayerData& layerData) noexcept
    {
        const Tensor& in = layerData.in;
        Tensor& out = layerData.out;

        const auto& weightsDims = weights.getDims();
        auto wInc = int(weightsDims[1]);
        auto inIt = in.begin();
        auto outIt = out.begin();
        MultiplyAddType multiplyAdd;

        auto weightsBegin = weights.begin();
        int its = int(weights.end() - weightsBegin) / wInc;

        for(auto wIt = weightsBegin, wEnd = weightsBegin + (its * wInc);
            wIt != wEnd; wIt += wInc)
        {
            *outIt += multiplyAdd(&*inIt, &*wIt, wInc);
            ++outIt;
        }
    }
}

std::unique_ptr<DenseLayer> DenseLayer::create(std::istream& stream)
{
    auto weights = Tensor::create(2, stream);

    if(! weights)
    {
        PT_LOG_ERROR << "Weights tensor parse failed" << std::endl;
        return std::unique_ptr<DenseLayer>();
    }

    auto biases = Tensor::create(1, stream);

    if(! biases)
    {
        PT_LOG_ERROR << "Biases tensor parse failed" << std::endl;
        return std::unique_ptr<DenseLayer>();
    }

    auto activation = ActivationLayer::create(stream);

    if(! activation)
    {
        PT_LOG_ERROR << "Activation layer parse failed" << std::endl;
        return std::unique_ptr<DenseLayer>();
    }

    return std::unique_ptr<DenseLayer>(new DenseLayer(std::move(*weights), std::move(*biases),
                                                      std::move(activation)));
}

bool DenseLayer::apply(LayerData& layerData) const
{
    const Tensor& in = layerData.in;
    const auto& iw = in.getDims();

    if(iw.size() != 1)
    {
        PT_LOG_ERROR << "Input tensor dims count must be 1" <<
                            " (input dims: " << VectorPrinter<std::size_t>{ iw } << ")" << std::endl;
        return false;
    }

    const auto& ww = _weights.getDims();

    if(iw[0] != ww[1])
    {
        PT_LOG_ERROR << "Input tensor dims[0] must be the same as weights dims[1]" <<
                            " (input dims: " << VectorPrinter<std::size_t>{ iw } << ")" <<
                            " (weights dims: " << VectorPrinter<std::size_t>{ ww } << ")" << std::endl;
        return false;
    }

    Tensor& out = layerData.out;
    _biases.copyTo(out);

    auto tensorSize = int(ww[1]);

    if(PT_LOOP_UNROLLING_ENABLE && tensorSize && tensorSize % (Tensor::VectorSize * 2) == 0)
    {
        multiplyAddImpl<Vector2MultiplyAdd>(_weights, layerData);
    }
    else if(tensorSize && tensorSize % Tensor::VectorSize == 0)
    {
        multiplyAddImpl<VectorMultiplyAdd>(_weights, layerData);
    }
    else
    {
        multiplyAddImpl<ScalarMultiplyAdd>(_weights, layerData);
    }

    _activation->apply(out);
    return true;
}

DenseLayer::DenseLayer(Tensor&& weights, Tensor&& biases,
                       std::unique_ptr<ActivationLayer>&& activation) noexcept :
    _weights(std::move(weights)),
    _biases(std::move(biases)),
    _activation(std::move(activation))
{
}

}
