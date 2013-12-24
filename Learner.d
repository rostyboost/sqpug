module Learner;

import std.conv;
import std.random;
import std.stdio;

import IO;


class Learner {

    private Random _gen;
    float[int] weights;

    private ulong _next_rnd_index(const ulong n)
    {
        return uniform(0, n, _gen);
    } 

    void learn(Observation[] data, float lambda)
    {
        ulong n = data.length;

        float[] dual_vars = new float[n];
        for(int i = 0; i < n; ++i)
            dual_vars[i] = 0;

        for(int i=0; i < 1000; ++i)
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
        }
    }

    float predict(Feature[] features)
    {
        float dotProd = 0;
        foreach(Feature feat; features)
        {
            if (! (feat[0] in this.weights))
                this.weights[feat[0]] = 0;
            dotProd += this.weights[feat[0]] * feat[1];
        }
        return dotProd;
    }


}
