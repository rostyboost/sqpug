module Learner;

import std.algorithm;
import std.math;
import std.conv;
import std.random;
import std.stdio;
import std.range;
import std.parallelism;

import Common;
import Hasher;
import IO;


class Learner {

    private Random _gen;
    float[] weights;
    uint bits;

    float intercept;

    private void delegate(ref Observation[], ref float[], ulong, float) _learn_internal;
    private float delegate(ref Observation[], ref float[], float) _duality_gap;

    float delegate(ref Feature[]) predict;
    void delegate(const ref Feature[], ref float[] scores) predict_multiclass;

    private LossType _lossType;

    // Multi-class specific variables
    private uint _n_classes;
    private float[] _a;
    private float[] _zeros;
    private float[] _mu_hat;
    private float[] _mu_bar;
    private float[] _mu;
    private float[] _z;
    private uint[] _multi_seeds;
    private float[] intercepts;

    this(const uint num_bits, const LossType loss, const uint n_classes)
    {
        _gen.seed(42);
        this.bits = num_bits;
        this.weights = new float[1 << bits];
        for(int i = 0; i < this.weights.length; ++i)
            this.weights[i] = 0;
        this.intercept = 0;

        _lossType = loss;
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
            case LossType.multiclass:
                _learn_internal = &learn_multiclass_svm;
                _duality_gap = &duality_gap_multiclass_svm;
                predict_multiclass = &predict_multiclass_svm;
                break;
            default:
                goto case LossType.squared;
        }

        // multiclass-specific initialization
        _n_classes = n_classes;
        if(n_classes > 2)
            this.init_multi();
    }

    private void init_multi()
    {
        for(int i = 0; i < this._n_classes; ++i)
        {
            char[] tmp = cast(char[])to!string(93 + i);
            this._multi_seeds[i] = Hasher.Hasher.MurmurHash3(tmp);
            this.intercepts[i] = 0;
            this._a[i] = 0;
            this._zeros[i] = 0;
            this._mu_hat[i] = 0;
            this._mu_bar[i] = 0;
            this._mu[i] = 0;
            this._z[i] = 0;
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
        float[] dual_vars;
        if(_lossType != LossType.multiclass)
            dual_vars = new float[n];
        else
            dual_vars = new float[n * this._n_classes];
        for(int i = 0; i < dual_vars.length; ++i)
            dual_vars[i] = 0;

        uint ind = 0;
        float delta_gap = 1;
        float last_gap = 1;
        float epsilon = 1e-3;
        auto gap_task = task(_duality_gap, data, dual_vars, lambda);
        bool gotResult = true;
        while(ind < n || delta_gap > epsilon)
        {
            if(gotResult && ind > n)
            {
                gap_task = task(_duality_gap, data, dual_vars, lambda);
                gap_task.executeInNewThread();
                gotResult = false;
            }
            ulong index = this._next_rnd_index(n);

            this._learn_internal(data, dual_vars, index, lambda);

            // stoping criterion: duality gap
            if(gap_task.done())
            {
                float gap = gap_task.yieldForce;
                delta_gap = 0.01 * delta_gap + 0.99 * abs((gap - last_gap)/last_gap);
                last_gap = gap;
                gotResult = true;
                stderr.writeln("Duality gap: ", gap);
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
            l2_norm_x += feat.val * feat.val;
        }

        float delta_dual = - (dual_vars[index] + pred - ex.label)/(
            1 + l2_norm_x/(2 * lambda * n));

        dual_vars[index] += delta_dual;

        foreach(Feature feat; ex.features)
        {
            this.weights[feat.id] += delta_dual * feat.val / (lambda * n);
        }
        this.intercept += delta_dual / (lambda * n);
    }

    float conjugate_logistic(float x)
    {
        if(x <= 0.0 || x >= 1.0)
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
            l2_norm_x += feat.val * feat.val;
        }

        float q = -1.0/(1.0 + exp(-p)) - dual_vars[index];

        float alpha = dual_vars[index];
        float phi_star = this.conjugate_logistic(-alpha);

        float delta_dual = q;
        if (abs(q) > 1e-20)
        {
            float log1expp = p;
            if (p <= 20)
                log1expp = log(1 + exp(p));
            delta_dual = q * min(
            1,
            (log1expp + phi_star + p * alpha + 2 * q *q)/(
               q * q * (4 + l2_norm_x/(lambda * n))));
        }
        dual_vars[index] += delta_dual;

        foreach(Feature feat; ex.features)
        {
            this.weights[feat.id] += -sgn * delta_dual * feat.val / (lambda * n);
        }
        this.intercept += -sgn * delta_dual / (lambda * n);
    }

    private bool optimize_dual_internal(ref float[] mu,
                                        float C,
                                        ref float[] result,
                                        ref float[] _mu_hat,
                                        ref float[] _mu_bar,
                                        ref float[] _z)
    {
        ulong k = mu.length;
        for(int j = 0; j < k; ++j)
            _mu_hat[j] = max(0, mu[j]);
        sort!("a > b")(_mu_hat);
        float s = 0;
        for(int j = 0; j < k; ++j)
        {
            s += _mu_hat[j];
            _mu_bar[j] = s;
            _z[j] = min(_mu_bar[j] - (j + 1) * _mu_hat[j], 1);
        }
        for(int j = 0; j < k; ++j)
        {
            float val = _mu_bar[j]/(1 + (j + 1) * C);
            if(val >= _z[j] && val <= _z[j + 1])
            {
                float treshold = (-val + _mu_bar[j]) / (j + 1);
                for(int i = 0; i < k; ++i)
                    result[i] = max(0, mu[i] - treshold);
                return true;
            }
        }
        ulong min_ind = k + 1;
        for(int j = 0; j < k; ++j)
            if(_z[j] == 1)
            {
                min_ind = j;
                break;
            }

        float norm_sq = 0;
        float diff_norm_sq = 0;
        float treshold = (-_z[min_ind] + _mu_bar[min_ind])/(min_ind + 1);
        for(int j = 0; j < k; ++j)
        {
            result[j] = max(0, mu[j] - treshold);
            norm_sq += mu[j] * mu[j];
            diff_norm_sq += (result[j] - mu[j]) * (result[j] - mu[j]);
        }
        if(diff_norm_sq + C < norm_sq)
            return true;
        return false;
    }

    private void learn_multiclass_svm(ref Observation[] data,
                                      ref float[] dual_vars,
                                      ulong index, float lambda)
    {
        ulong n = data.length;
        Observation ex = data[index];
        //TODO
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
            float p = -(2*data[i].label - 1) * this.dotProd(data[i].features);
            float conj = this.conjugate_logistic(-dual_vars[i]);
            float log1expp = p;
            if(p <= 20)
                log1expp = log(1.0 + exp(p));
            gap += log1expp + conj;
        }
        gap /= n;
        gap += lambda * this.normsq_weights();

        return gap;
    }

    private float duality_gap_multiclass_svm(ref Observation[] data,
                                             ref float[] dual_vars,
                                             float lambda)
    {
        return 0;
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
        float dp = this.intercept;
        foreach(Feature feat; features)
            dp += this.weights[feat.id] * feat.val;
        return dp;
    }

    float predict_logistic(ref Feature[] features)
    {
        float dp = this.dotProd(features);
        return 1.0 / ( 1.0 + exp(-dp));
    }

    void predict_multiclass_svm(const ref Feature[] features,
                                ref float[] scores)
    {

    }

}
