module Serializer;

import std.array;
import std.conv;
import std.stdio;
import std.string;
import std.typecons;

import Common;
import Learner;

void dump_model(const Learner model, const Options opts, const string path)
{
    auto f = File(path, "w");

    f.writeln("bits:" ~ to!string(opts.bits));
    f.writeln("loss:" ~ to!string(opts.loss));
    f.writeln("intercept:" ~ to!string(model.intercept));
    for(int i = 0; i < model.weights.length; ++i)
        f.writeln(to!string(i) ~ ":" ~ to!string(model.weights[i]));
    f.close();
}

Learner load_model(const string model_path)
{
    auto f = File(model_path, "r");

    string token = split(f.readln(), ":")[1];
    uint bits = to!uint(stripRight(token));

    token = split(f.readln(), ":")[1];
    LossType loss = to!LossType(stripRight(token));

    token = split(f.readln(), ":")[1];
    float intercept = to!float(stripRight(token));

    Learner model = new Learner(bits);
    model.intercept = intercept;

    string line;
    while ((line = f.readln()) !is null)
    {
        auto tokens = split(line, ":");
        uint ind = to!uint(tokens[0]);
        float w = to!float(stripRight(tokens[1]));

        model.weights[ind] = w;
    }
    f.close();

    return model;
}
