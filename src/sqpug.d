import std.conv;
import std.getopt;
import std.stdio;

import IO;
import Learner;

struct Options {
    string data; // input data path
    ushort bits; // number of bits for hashing trick
}


void main(string[] args) {

    Options opts;

    getopt(
        args,
        "data", &opts.data,
        "bits", &opts.bits);

    InMemoryData data = new InMemoryData(opts.data, opts.bits);

    Learner learner = new Learner();
    learner.learn(data.data, 0.25);
    
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

