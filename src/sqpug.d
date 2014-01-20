import std.conv;
import std.getopt;
import std.stdio;

import Common;
import IO;
import Learner;
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
        "model_in", &opts.model_in);

    Learner learner;
    if(opts.model_in == "")
    {
        InMemoryData data = new InMemoryData(opts.data, opts.bits);

        learner = new Learner(opts.bits);
        learner.learn(data.data, opts.lambda);
    }
    else
        learner = Serializer.load_model(opts.model_in);

    if(opts.model_out != "")
        Serializer.dump_model(learner, opts, opts.model_out);

    InMemoryData test_data = new InMemoryData(opts.test, opts.bits);

    float error = 0;
    foreach(Observation obs; test_data)
    {
        float pred = learner.predict(obs.features);
        error += (pred - obs.label) * (pred - obs.label);
        writeln(to!string(obs.label) ~ ";" ~ to!string(pred));
    }
    error /= test_data.data.length;

    writeln("Total error: " ~ to!string(error));
}

