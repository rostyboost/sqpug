import std.conv;
import std.stdio;

import IO;
import Learner;


void main() {
    writeln("Hello World!");
    Observation[] data = IO.load_data("train.txt");

    Learner learner = new Learner();
    learner.learn(data, 3);
    
    Observation[] test_data = IO.load_data("test.txt");

    foreach(Observation obs; test_data)
    {
        float pred = learner.predict(obs.features);
        writeln(to!string(obs.label) ~ ";" ~ to!string(pred));
    }
}
