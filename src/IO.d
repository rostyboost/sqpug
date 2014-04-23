module IO;

import core.atomic;
import std.array;
import std.conv;
import std.parallelism;
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

interface IData {
    // Basic InputRange<Observation> functions
    bool empty();
    void popFront();
    Observation front();

    // Array like stuff. TODO: consider RandomInfinite Range instead
    Observation opIndex(size_t i);
    @property ulong length();

    void rewind();
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

    public const int counter()
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

class StreamData : IData {

    private ulong _cnt;
    private bool _finished;
    private File _f;
    private bool _isStdIn;

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

    int _numFeatures;


    this(const string file_path, int bits)
    {
        bitMask = (1 << bits) - 1;

        _f = stdin;
        _isStdIn = true;
        if (file_path != "")
        {
            _f = File(file_path, "r");
            _isStdIn = false;
        }

        this._initializeState();
    }

    ~ this()
    {
        _f.close();
    }

    void _initializeState()
    {
        _finished = false;
        _cnt = 0;

        buffer = bufferA;
        last_buffer = bufferB;
        current_features.length = 100;
        _numFeatures = 0;

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

    bool empty()
    {
        return _finished;
    }

    private void _loadBuffer()
    {
        if(num_buff % 2 == 0)
            buffer = bufferA;
        else
            buffer = bufferB;
        buffer[] = '\t';
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
                _numFeatures = 0;
                break;
            case '\n':
                label_start = _indBuffer + 1;
                if(label_start == buff_size)
                    label_start = 0;
                //Last feature value:
                val_end = _indBuffer;
                feat_val = to_float(get_slice(val_start, val_end));
                if(_numFeatures == current_features.length)
                    current_features.length *= 2;
                current_features[_numFeatures] = Feature(feat_hash, feat_val);
                _numFeatures++;
                current_features.length = _numFeatures;
                //Current example is ready:
                _currentObs = Observation(label, current_features);
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
                if(_numFeatures == current_features.length)
                    current_features.length *= 2;
                current_features[_numFeatures] = Feature(feat_hash, feat_val);
                _numFeatures++;
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

    Observation opIndex(size_t i) { return _currentObs; } // TODO: throw exception

    @property ulong length() { return _cnt; }

    void rewind()
    {
        if(_isStdIn)
            throw new Exception("stdin stream can't be rewound.");
        _f.rewind();
        this._initializeState();
    }
}

class ThreadedStreamData : IData {

    private StreamData _stream;
    private ConcurrentQueue _queue;
    private Observation _currentObs;
    Task!(run, void delegate(ref StreamData, ref ConcurrentQueue),
          StreamData, ConcurrentQueue)* _taskFillQ;

    this(const string file_path, uint bits)
    {
        _stream = new StreamData(file_path, bits);
        _queue = new ConcurrentQueue(100);

        // Start reading the stream in a thread
        _taskFillQ = task(&fillQ, _stream, _queue);
        _taskFillQ.executeInNewThread();
        popFront();
    }

    private void fillQ(ref StreamData stream, ref ConcurrentQueue queue)
    {
        foreach(Observation obs; stream)
            while(!queue.push(obs)){}
        queue.done_inserting = true;
    }

    bool empty()
    {
        return _taskFillQ.done() && _queue.counter() == 0;
    }

    void popFront()
    {
        while(!_queue.pop(_currentObs))
        {
            if(empty())
                return;
        }
    }

    Observation front()
    {
        return _currentObs;
    }

    Observation opIndex(size_t i) { return _currentObs; } // TODO: throw exception

    @property ulong length() { return _stream.length; }

    void rewind()
    {
        _stream.rewind();

        _queue = new ConcurrentQueue(100);

        // Start reading the stream in a thread
        _taskFillQ = task(&fillQ, _stream, _queue);
        _taskFillQ.executeInNewThread();
        popFront();
    }
}

class InMemoryData : IData {

    Observation[] data;

    private ulong _current_cnt;

    this(const string file_path, const ref Options opts)
    {
        auto stream = new StreamData(file_path, opts.bits);

        foreach(Observation obs; stream)
        {
            Feature[] cp_feats = obs.features.dup;
            data ~= Observation(obs.label, cp_feats);
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

    Observation opIndex(size_t i) { return data[i]; }

    @property ulong length() { return data.length; }

    void rewind()
    {
        this._current_cnt = 0;
    }
}
