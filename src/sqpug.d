import std.conv;
import std.getopt;
import std.stdio;
import std.random;

import Common;
import IO;
import Learner;
import PredictionServer;
import Serializer;


void main(string[] args) {

    Options opts;

    getopt(
        args,
        "data", &opts.data,
        "passes", &opts.passes,
        "bits", &opts.bits,
        "loss", &opts.loss,
        "lambda", &opts.lambda,
        "test", &opts.test,
        "model_out", &opts.model_out,
        "model_in", &opts.model_in,
        "format", &opts.data_format,
        "n_classes", &opts.n_classes,
        "server", &opts.server_mod,
        "port", &opts.server_port);

    Learner learner;
    if(opts.model_in == "")
    {
        IData data = new InMemoryData(opts.data, opts);

        learner = new Learner(opts.bits, opts.loss, opts.n_classes);
        learner.learn(data, opts.passes, opts.lambda);
    }
    else
        learner = Serializer.load_model(opts.model_in);

    if(opts.model_out != "")
        Serializer.dump_model(learner, opts, opts.model_out);

    if(opts.server_mod)
    {
        PredictionServer server = new PredictionServer(opts.server_port,
                                                       learner);
        server.serve_forever();
    }

    if(opts.test != "")
    {
        IData test_data = new StreamData(opts.test, opts.bits);

        float error = 0;
        float baseline_error = 0;
        if(opts.loss == LossType.squared)
        {
            foreach(Observation obs; test_data)
            {
                float pred = learner.predict(obs.features);
                //stdout.writeln(pred);
                error += (pred - obs.label) * (pred - obs.label);
            }
        }
        else
        {
            Random gen;
            gen.seed(42);
            float av_pred = 0;
            float av_label = 0;
            long num_pos = 0;
            long num_neg = 0;
            foreach(Observation obs; test_data)
            {
                if(obs.label == 1.0)
                {
                    av_label += 1;
                    num_pos += 1;
                }
                else
                    num_neg += 1;
                av_pred += learner.predict(obs.features);
            }
            av_pred /= test_data.length;
            av_label /= test_data.length;
            stderr.writeln("Average pred: ", av_pred);
            stderr.writeln("Average label: ", av_label);
            stderr.writeln(num_pos, " positives, ", num_neg, " negatives.");
            test_data.rewind();
            foreach(Observation obs; test_data)
            {
                float pred = learner.predict(obs.features);
                //stdout.writeln(pred);
                float pred_label = 1;
                if(pred < av_label)
                    pred_label = -1;
                if(pred_label * (2 * obs.label -1) < 0)
                    error += 1;

                float base_pred = -1;
                if(uniform(0.0f, 1.0f, gen) < av_label)
                    base_pred = 1;
                if(base_pred * (2 * obs.label -1) < 0)
                    baseline_error += 1;
            }
        }
        error /= test_data.length;
        baseline_error /= test_data.length;

        stderr.writeln("Total error: ", error);
        stderr.writeln("Baseline error: ", baseline_error);
    }
}

