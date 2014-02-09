import std.conv;
import std.getopt;
import std.stdio;

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
        InMemoryData data = new InMemoryData(opts.data, opts);

        learner = new Learner(opts.bits, opts.loss, opts.n_classes);
        learner.learn(data.data, opts.lambda);
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
        InMemoryData test_data = new InMemoryData(opts.test, opts);

        float error = 0;
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
            foreach(Observation obs; test_data)
            {
                float pred = learner.predict(obs.features);
                float pred_label = 1;
                if(pred < 0.5)
                    pred_label = -1;
                //stdout.writeln(pred);
                if(pred_label * (2 * obs.label -1) < 0)
                    error += 1;
            }
        }
        error /= test_data.data.length;

        stderr.writeln("Total error: ", error);
    }
}

