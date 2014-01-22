module Learner;

import std.math;
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
        _gen.seed(42);
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
        stderr.writeln("Starting learning on ", n, " datapoints.");
        float[] dual_vars = new float[n];
        for(int i = 0; i < n; ++i)
            dual_vars[i] = 0;

        uint ind = 0;
        float delta_gap = 1;
        float last_gap = 1;
        float epsilon = 1e-6;
        while( ind < n || delta_gap > epsilon)
        //for(int i=0; i < 1000; ++i)
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

            // stoping criterion: duality gap
            if(ind > n && ind % n/5 == 0)
            {
                float gap = this._duality_gap(data, dual_vars, lambda);
                delta_gap = 0.8 * delta_gap + 0.2 * abs((gap - last_gap)/last_gap);
                last_gap = gap;
            }
            ind += 1;
        }
        stderr.writeln("Stopped SDCA after ", ind, " sampled points.");
    }

    private float _duality_gap(ref Observation[] data,
                               ref float[] dual_vars,
                               const float lambda)
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

        float norm = 0;
        foreach(float w; this.weights)
            norm += w * w;
        gap += lambda * norm;

        return gap;
    }

    final float predict(ref Feature[] features)
    {
        float dotProd = this.intercept;
        foreach(Feature feat; features)
            dotProd += this.weights[feat[0]] * feat[1];
        return dotProd;
    }


}
