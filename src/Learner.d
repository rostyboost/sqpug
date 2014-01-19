module Learner;

import std.conv;
import std.random;
import std.stdio;
import std.range;

import IO;


class Learner {

    private Random _gen;
    float[] weights;

    float intercept;

    this(const uint bits)
    {
        this.weights = new float[1 << bits];
        for(int i = 0; i < this.weights.length; ++i)
            this.weights[i] = 0;
        this.intercept = 0;
    }

    private ulong _next_rnd_index(const ulong n)
    {
        return uniform(0, n, _gen);
    } 

    void learn(ref Observation[] data, const float lambda)
    {
        ulong n = data.length;

        float[] dual_vars = new float[n];
        for(int i = 0; i < n; ++i)
            dual_vars[i] = 0;

        for(int i=0; i < 2_000_000; ++i)
        {
            ulong index = this._next_rnd_index(n);
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
    }

    float predict(ref Feature[] features)
    {
        float dotProd = this.intercept;
        foreach(Feature feat; features)
            dotProd += this.weights[feat[0]] * feat[1];
        return dotProd;
    }


}
