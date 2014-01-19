import std.conv;
import std.getopt;
import std.stdio;

import Common;
import IO;
import Learner;


void main(string[] args) {

    Options opts;

    getopt(
        args,
        "data", &opts.data,
        "bits", &opts.bits,
        "loss", &opts.loss,
        "lambda", &opts.lambda,
        "test", &opts.test);

    InMemoryData data = new InMemoryData(opts.data, opts.bits);

    Learner learner = new Learner(opts.bits);
    learner.learn(data.data, opts.lambda);

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

