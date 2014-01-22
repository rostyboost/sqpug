module Learner;

import std.algorithm;
import std.math;
import std.conv;
import std.random;
import std.stdio;
import std.range;
import std.parallelism;

import Common;
import IO;


class Learner {

    private Random _gen;
    float[] weights;

    float intercept;

    private void delegate(ref Observation[], ref float[], ulong, float) _learn_internal;
    private float delegate(ref Observation[], ref float[], float) _duality_gap;

    float delegate(ref Feature[]) predict;


    this(const uint bits, const LossType loss)
    {
        _gen.seed(42);
        this.weights = new float[1 << bits];
        for(int i = 0; i < this.weights.length; ++i)
            this.weights[i] = 0;
        this.intercept = 0;

        switch(loss)
        {
            case LossType.squared:
                _learn_internal = &learn_squared;
                _duality_gap = &duality_gap_squared;
                predict = &dotProd;
                break;
            case LossType.logistic:
                _learn_internal = &learn_logistic;
                _duality_gap = &duality_gap_logistic;
                predict = &predict_logistic;
                break;
            default:
                _learn_internal = &learn_squared;
                _duality_gap = &duality_gap_squared;
                predict = &dotProd;
        }
    }

    private ulong _next_rnd_index(const ulong n)
    {
        return uniform(0, n, _gen);
    }

    public void learn(ref Observation[] data, float lambda)
    {
        ulong n = data.length;
        stderr.writeln("Starting learning on ", n, " datapoints.");
        float[] dual_vars = new float[n];
        for(int i = 0; i < n; ++i)
            dual_vars[i] = 0;

        uint ind = 0;
        float delta_gap = 1;
        float last_gap = 1;
        float epsilon = 1e-8;
        auto gap_task = task(_duality_gap, data, dual_vars, lambda);
        bool gotResult = true;
        float div = 5;
        while( ind < n || delta_gap > epsilon)
        {
            if(gotResult && ind > n && ind % to!int(n/div) == 0)
            {
                gap_task = task(_duality_gap, data, dual_vars, lambda);
                gap_task.executeInNewThread();
                gotResult = false;
                div *= 2;
            }
            ulong index = this._next_rnd_index(n);

            this._learn_internal(data, dual_vars, index, lambda);

            // stoping criterion: duality gap
            if(gap_task.done())
            {
                float gap = gap_task.yieldForce;
                delta_gap = 0.8 * delta_gap + 0.2 * abs((gap - last_gap)/last_gap);
                last_gap = gap;
                gotResult = true;
            }
            ind += 1;
        }
        stderr.writeln("Stopped SDCA after ", ind, " sampled points.");
    }

    private void learn_squared(ref Observation[] data,
                               ref float[] dual_vars,
                               ulong index, float lambda)
    {
        ulong n = data.length;
        Observation ex = data[index];

        float pred = this.predict(ex.features);
        float l2_norm_x = 0; // TODO: merge with former line
        foreach(Feature feat; ex.features)
        {
            l2_norm_x += feat[1] * feat[1];
        }

        float delta_dual = - (dual_vars[index] + pred - ex.label)/(
            1 + l2_norm_x/(2 * lambda * n));

        dual_vars[index] += delta_dual;

        foreach(Feature feat; ex.features)
        {
            this.weights[feat[0]] += delta_dual * feat[1] / (lambda * n);
        }
        this.intercept += delta_dual / (lambda * n);
    }

    float conjugate_logistic(float x)
    {
        if(x == 0.0)
            return 0;
        return x * log(x) + (1 - x) * log(1.0 - x);
    }

    private void learn_logistic(ref Observation[] data,
                                ref float[] dual_vars,
                                ulong index, float lambda)
    {
        ulong n = data.length;
        Observation ex = data[index];

        float p = -this.dotProd(ex.features);
        float sgn = 1.0;
        if(ex.label == 0.0)
            sgn = -1.0;
        p = sgn * p;
        float l2_norm_x = 0; // TODO: merge with former line
        foreach(Feature feat; ex.features)
        {
            l2_norm_x += feat[1] * feat[1];
        }

        float q = -1.0/(1.0 + exp(-p)) - dual_vars[index];

        float alpha = dual_vars[index];
        float phi_star = this.conjugate_logistic(-alpha);

        float delta_dual = 1;
        if (q != 0)
            delta_dual = q * min(
            1,
            (log(1 + exp(p)) + phi_star + p * alpha + 2 * q *q)/(
               q * q * (4 + l2_norm_x/(lambda * n))));
        dual_vars[index] += delta_dual;

        foreach(Feature feat; ex.features)
        {
            this.weights[feat[0]] += -sgn * delta_dual * feat[1] / (lambda * n);
        }
        this.intercept += -sgn * delta_dual / (lambda * n);
    }

    private float duality_gap_squared(ref Observation[] data,
                                      ref float[] dual_vars,
                                      float lambda)
    {
        float gap = 0;
        ulong n = data.length;
        for(int i = 0; i < n; ++i)
        {
            float pred = this.predict(data[i].features);
            gap += (pred - data[i].label) * (pred - data[i].label);
            gap += (dual_vars[i] * dual_vars[i]
                    - 2 * dual_vars[i] * data[i].label);
        }
        gap /= (2 * n);

        gap += lambda * this.normsq_weights();

        return gap;
    }

    private float duality_gap_logistic(ref Observation[] data,
                                       ref float[] dual_vars,
                                       float lambda)
    {
        float gap = 0;
        ulong n = data.length;
        for(int i = 0; i < n; ++i)
        {
            float p = (2*data[i].label - 1) * this.dotProd(data[i].features);
            float conj = this.conjugate_logistic(-dual_vars[i]);
            gap += log(1.0 + exp(p)) + conj;
        }
        gap /= n;

        gap += lambda * this.normsq_weights();

        return gap;
    }

    float normsq_weights()
    {
        float normsq = 0;
        foreach(float w; this.weights)
            normsq += w * w;
        return normsq;
    }

    float dotProd(ref Feature[] features)
    {
        float dotProd = this.intercept;
        foreach(Feature feat; features)
            dotProd += this.weights[feat[0]] * feat[1];
        return dotProd;
    }

    float predict_logistic(ref Feature[] features)
    {
        float dotProd = this.dotProd(features);
        return 1.0 / ( 1.0 + exp(-dotProd));
    }

}
