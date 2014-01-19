import std.conv;
import std.getopt;
import std.stdio;

import IO;
import Learner;

enum LossType {
    squared = "squared",
    logistic = "logistic"
}

struct Options {
    string data; // input data path
    ushort bits; // number of bits for hashing trick
    LossType loss;
    float lambda;
}


void main(string[] args) {

    Options opts;

    getopt(
        args,
        "data", &opts.data,
        "bits", &opts.bits,
        "loss", &opts.loss,
        "lambda", &opts.lambda);

    InMemoryData data = new InMemoryData(opts.data, opts.bits);

    Learner learner = new Learner();
    learner.learn(data.data, opts.lambda);
    
    InMemoryData test_data = new InMemoryData("test_data/example_test.txt",
                                              opts.bits);


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

