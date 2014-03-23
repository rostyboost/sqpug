module IO;

import core.atomic;
import std.array;
import std.conv;
import std.stdio;

import Common;
import Hasher;

struct Observation {

    float label;
    Feature[] features;

    this(float label_, ref Feature[] features_)
    {
        label = label_;
        features = features_;
    }
}

class ConcurrentQueue {
// Thread-safe concurrent queue. Only safe for 1 producer, 1 consumer.
// But lock-free, so should be fast.
    Observation[] _buffer;
    int _head;
    int _tail;
    int _size;
    shared(int) _cnt;
    bool done_inserting;

    this(const uint size)
    {
        _head = 0;
        _tail = -1;
        _size = size;
        done_inserting = false;
        _buffer = new Observation[size];
    }

    public int counter()
    {
        return _cnt;
    }

    public int capacity()
    {
        return _size;
    }

    public bool push(Observation obs)
    {
        if(_cnt == _size)
            return false;
        ++_tail;
        if(_tail == _size)
            _tail = 0;
        _buffer[_tail] = obs;

        // increment _cnt
        while (!cas(&_cnt, _cnt, _cnt + 1)){};

        return true;
    }

    public bool pop(ref Observation obs)
    {
        if(_cnt == 0)
            return false;
        obs = _buffer[_head];
        ++_head;
        if(_head == _size)
            _head = 0;

        // decrement _cnt
        while (!cas(&_cnt, _cnt, _cnt - 1)){};

        return true;
    }

}

class StreamData {

    private ulong _cnt;
    private bool _finished;
    private File _f;

    Observation _currentObs;

    uint bitMask;

    uint buff_size = 20_000;

    char[20_000] bufferA;
    char[20_000] bufferB;
    char[40_000] tmp_split_buff;
    char[] buffer;
    char[] last_buffer;
    Feature[] current_features;

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

    int feat_start;
    int feat_end;
    uint feat_hash;

    int val_start;
    int val_end;
    float feat_val;

    int label_start;
    int label_end;
    float label;

    uint num_buff;
    int _indBuffer;


    this(const string file_path, int bits)
    {
        bitMask = (1 << bits) - 1;

        _f = stdin;
        if (file_path != "")
            _f = File(file_path, "r");

        _finished = false;
        _cnt = 0;

        buffer = bufferA;
        last_buffer = bufferB;
        current_features.length = 0;

        feat_start = 0;
        feat_end = 0;
        feat_hash = -1;

        val_start = 0;
        val_end = 0;
        feat_val = -1;

        label_start = 0;
        label_end = 0;
        label = -1;

        num_buff = 0;
        _indBuffer = 0;

        this._loadBuffer();
        this.popFront();
    }

    bool empty() const
    {
        return _finished;
    }

    private void _loadBuffer()
    {
        if(num_buff % 2 == 0)
            buffer = bufferA;
        else
            buffer = bufferB;
        for(int i = 0; i < buff_size; ++i)
            buffer[i] = '\t';
        _f.rawRead(buffer);
        _indBuffer = 0;
    }

    private bool _processToken()
    {
        switch(buffer[_indBuffer])
        {
            case '|':
                label_end = _indBuffer;
                feat_start = _indBuffer + 1;
                if(feat_start == buff_size)
                    feat_start = 0;
                label = to_float(get_slice(label_start, label_end));
                break;
            case '\n':
                label_start = _indBuffer + 1;
                if(label_start == buff_size)
                    label_start = 0;
                //Last feature value:
                val_end = _indBuffer;
                feat_val = to_float(get_slice(val_start, val_end));
                current_features ~= Feature(feat_hash, feat_val);
                //Current example is ready:
                _currentObs = Observation(label, current_features);
                current_features = new Feature[0];
                return true;
                break;
            case ':':
                feat_end = _indBuffer;
                val_start = _indBuffer+1;
                if(val_start == buff_size)
                    val_start = 0;
                feat_hash = Hasher.Hasher.MurmurHash3(
                    get_slice(feat_start, feat_end)) & bitMask;
                break;
            case ' ':
                val_end = _indBuffer;
                feat_start = _indBuffer + 1;
                if(feat_start == buff_size)
                    feat_start = 0;
                feat_val = to_float(get_slice(val_start, val_end));
                current_features ~= Feature(feat_hash, feat_val);
                break;
            default:
                break;
        }
        return false;
    }

    void popFront()
    {
        bool nextReady = false;
        while(!nextReady)
        {
            while(_indBuffer < buff_size)
            {
                nextReady = _processToken();
                _indBuffer++;
                if(nextReady)
                {
                    ++_cnt;
                    return;
                }
            }
            if(_f.eof && _indBuffer == buff_size)
            {
                _finished = true;
                _f.close();
                break;
            }
            last_buffer = buffer;
            num_buff++;
            this._loadBuffer();
        }
    }

    Observation front()
    {
        return _currentObs;
    }

}

class InMemoryData {

    Observation[] data;

    private ulong _current_cnt;

    this(const string file_path, const ref Options opts)
    {
        StreamData stream = new StreamData(file_path, opts.bits);

        foreach(Observation obs; stream)
        {
            data ~= obs;
        }

        this._current_cnt = 0;
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
