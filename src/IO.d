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

        char[] bufferA = new char[buff_size];
        char[] bufferB = new char[buff_size];
        char[] tmp_split_buff = new char[2 * buff_size];
        char[] buffer = bufferA;
        char[] last_buffer = bufferB;
        Feature[] current_features = new Feature[0];

        int fill_split_buff(int ind_start, int ind_end)
        {
            int size_end = buff_size - ind_start;
            int size_start = ind_end;
            tmp_split_buff[0..size_end] = (
                last_buffer[ind_start..buff_size]);
            tmp_split_buff[size_end..size_end+size_start]=(
                buffer[0..ind_end]);
            return size_end + size_start;
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

        bool new_line = true;
        uint num_buff = 0;
        bool dump_last = false;
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
                if(new_line)
                {
                    label_start = ind;
                    new_line =false;
                }
                switch(buffer[ind])
                {
                    case '|':
                        label_end = ind;
                        feat_start = ind + 1;
                        if(feat_start == buff_size)
                            feat_start = 0;
                        if(label_start < label_end)
                        {
                            label = to!float(buffer[label_start..label_end]);
                        }
                        else
                        {
                            int ind_end = fill_split_buff(label_start, label_end);
                            label = to!float(tmp_split_buff[0..ind_end]);
                        }
                        break;
                    case '\n':
                        new_line = true;
                        label_start = ind+1;
                        if(label_start == buff_size)
                            label_start = 0;
                        dump_last = true;
                        break;
                    case ':':
                        feat_end = ind;
                        val_start = ind+1;
                        if(val_start == buff_size)
                            val_start = 0;
                        if(feat_start < feat_end)
                        {
                            feat_hash = Hasher.Hasher.MurmurHash3(
                                buffer[feat_start..feat_end]) & bitMask;
                        }
                        else
                        {
                            int ind_end = fill_split_buff(feat_start, feat_end);
                            feat_hash = Hasher.Hasher.MurmurHash3(
                                tmp_split_buff[0..ind_end]) & bitMask;
                        }
                        break;
                    case ' ':
                        val_end = ind;
                        feat_start = ind + 1;
                        if(feat_start == buff_size)
                            feat_start = 0;

                        if(val_start < val_end)
                        {
                            feat_val = to!float(buffer[val_start..val_end]);
                        }
                        else
                        {
                            int ind_end = fill_split_buff(val_start, val_end);
                            feat_val = to!float(tmp_split_buff[0..ind_end]);
                        }
                        current_features ~= Feature(feat_hash, feat_val);
                        break;
                    default: // eof
                        break;
                }
                if(dump_last)
                {
                    //Dump current example into dataset:
                    data ~= new Observation(label, current_features);
                    current_features = new Feature[0];
                    dump_last = false;
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
