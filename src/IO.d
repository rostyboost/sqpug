module IO;

import std.array;
import std.conv;
import std.stdio;
import std.typecons;

import Common;
import Hasher;

alias Tuple!(uint, float) Feature;

class Observation {

    float label;
    Feature[] features;

    this(float label_, ref Feature[] features_)
    {
        label = label_;
        features = features_;
    }
}

class InMemoryData {

    Observation[] data;

    private ulong _current_cnt;

    this(const string file_path, const ref Options opts)
    {
        File f = stdin;
        if (file_path != "")
            f = File(file_path, "r");

        if(opts.data_format == DataFormat.sparse)
            this.data = this.load_sparse(f, opts.bits);
        else if(opts.data_format == DataFormat.dense)
            this.data = this.load_dense(f);

        if (file_path != "")
            f.close();

        this._current_cnt = 0;
    }

    Observation[] load_sparse(ref File f, const uint bits)
    {
        Observation[] data;
        uint bitMask = (1 << bits) - 1;

        uint buff_size = 20000;

        char[20_000] bufferA;
        char[20_000] bufferB;
        char[40_000] tmp_split_buff;
        char[] buffer = bufferA;
        char[] last_buffer = bufferB;
        Feature[] current_features = new Feature[0];

        char[] get_slice(int ind_start, int ind_end)
        {
            if(ind_start < ind_end)
            {
                return buffer[ind_start..ind_end];
            }
            else // token needs to be reconstructed from the 2 buffers
            {
                int size_end = buff_size - ind_start;
                int size_start = ind_end;
                tmp_split_buff[0..size_end] = (
                    last_buffer[ind_start..buff_size]);
                tmp_split_buff[size_end..size_end+size_start]=(
                    buffer[0..ind_end]);
                return tmp_split_buff[0..size_end + size_start];
            }
        }

        int feat_start = 0;
        int feat_end = 0;
        uint feat_hash = -1;

        int val_start = 0;
        int val_end = 0;
        float feat_val = -1;

        int label_start = 0;
        int label_end = 0;
        float label = -1;

        uint num_buff = 0;
        while(!f.eof)
        {
            if(num_buff % 2 == 0)
                buffer = bufferA;
            else
                buffer = bufferB;
            for(int i = 0; i < buff_size; ++i)
                buffer[i] = '\t';
            f.rawRead(buffer);

            int ind = 0;
            while(ind < buff_size)
            {
                switch(buffer[ind])
                {
                    case '|':
                        label_end = ind;
                        feat_start = ind + 1;
                        if(feat_start == buff_size)
                            feat_start = 0;
                        label = to_float(get_slice(label_start, label_end));
                        break;
                    case '\n':
                        label_start = ind+1;
                        if(label_start == buff_size)
                            label_start = 0;
                        //Dump current example into dataset:
                        data ~= new Observation(label, current_features);
                        current_features = new Feature[0];
                        break;
                    case ':':
                        feat_end = ind;
                        val_start = ind+1;
                        if(val_start == buff_size)
                            val_start = 0;
                        feat_hash = Hasher.Hasher.MurmurHash3(
                            get_slice(feat_start, feat_end)) & bitMask;
                        break;
                    case ' ':
                        val_end = ind;
                        feat_start = ind + 1;
                        if(feat_start == buff_size)
                            feat_start = 0;
                        feat_val = to_float(get_slice(val_start, val_end));
                        current_features ~= Feature(feat_hash, feat_val);
                        break;
                    default: // eof
                        break;
                }
                ind++;
            }
            last_buffer = buffer;
            num_buff++;
            if(f.eof)
                break;
        }
        return data;
    }

    Observation[] load_sparse_naive(ref File f, const uint bits)
    {
        Observation[] data;
        uint bitMask = (1 << bits) - 1;

        foreach (char[] line; lines(f))
        {
            auto tokens = split(line, "|");
            float label = to!float(tokens[0]);

            auto feats_tokens = split(tokens[1]);
            Feature[] features;
            foreach(char[] token; feats_tokens)
            {
                auto str_tuple = split(token, ":");
                uint feature_hash = Hasher.Hasher.MurmurHash3(str_tuple[0]);
                feature_hash = feature_hash & bitMask;
                features ~= Feature(feature_hash, to!float(str_tuple[1]));
            }
            data ~= new Observation(label, features);
        }
        return data;
    }

    Observation[] load_dense(ref File f)
    {
        Observation[] data;
        // TODO
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
