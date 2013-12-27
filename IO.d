module IO;

import std.array;
import std.conv;
import std.stdio;
import std.typecons;

alias Tuple!(int, float) Feature;

class Observation {

    float label;
    Feature[] features;

    this(float label_, Feature[] features_)
    {
        label = label_;
        features = features_;
    }
}

class InMemoryData {

    Observation[] data;

    private ulong _current_cnt;

    this(const string file_path)
    {
        this.data = this.load_data(file_path);
        this._current_cnt = 0;
    }

    Observation[] load_data(const string file_path)
    {
        auto f = File(file_path, "r"); // open for reading

        Observation[] data;

        foreach (char[] line; lines(f))
        {
            debug writeln(line);
            auto tokens = split(line, " | ");
            float label = to!float(tokens[0]);

            auto feats_tokens = split(tokens[1]);
            Feature[] features;
            foreach(char[] token; feats_tokens)
            {
                auto str_tuple = split(token, ":");
                features ~= Feature(to!int(str_tuple[0]),
                                    to!float(str_tuple[1]));
            }

            data ~= new Observation(label, features);
        }

        return data;
    }

    bool empty() const
    {
        return _current_cnt == data.length;
    }

    void popFront()
    {
        ++_current_cnt;
    }

    Observation front()
    {
        return data[_current_cnt];
    }
}
