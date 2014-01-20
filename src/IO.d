module IO;

import std.array;
import std.conv;
import std.stdio;
import std.typecons;

import Hasher;

alias Tuple!(uint, float) Feature;

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
    private uint _bitMask;

    this(const string file_path, const ushort bits)
    {
        this._bitMask = (1 << bits);
        this.data = this.load_data(file_path);
        this._current_cnt = 0;
    }

    Observation[] load_data(const string file_path)
    {
        auto f = stdin;
        if (file_path != "")
            f = File(file_path, "r");

        Observation[] data;

        foreach (char[] line; lines(f))
        {
            auto tokens = split(line, " | ");
            float label = to!float(tokens[0]);

            auto feats_tokens = split(tokens[1]);
            Feature[] features;
            foreach(char[] token; feats_tokens)
            {
                auto str_tuple = split(token, ":");
                uint feature_hash = Hasher.Hasher.MurmurHash3(str_tuple[0]);
                feature_hash = feature_hash % this._bitMask; //TODO: force D bit masking...
                features ~= Feature(feature_hash, to!float(str_tuple[1]));
            }
            data ~= new Observation(label, features);
        }
        if (file_path != "")
            f.close();
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
