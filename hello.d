import std.conv;
import std.stdio;

import IO;
import Learner;


void main() {
    writeln("Hello World!");
    Observation[] data = IO.load_data("test_data/example_train.txt");

    Learner learner = new Learner();
    learner.learn(data, 0.1);
    
    Observation[] test_data = IO.load_data("test_data/example_test.txt");


    float error = 0;

    foreach(Observation obs; test_data)
    {
        float pred = learner.predict(obs.features);
        error += (pred - obs.label) * (pred - obs.label);
        writeln(to!string(obs.label) ~ ";" ~ to!string(pred));
    }
    error /= test_data.length;

    writeln("Total error: " ~ to!string(error));
}
