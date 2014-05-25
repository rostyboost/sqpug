module Common;

import std.ascii;
import std.typecons;

struct Feature { uint id; float val;}

enum LossType {
    squared = "squared",
    logistic = "logistic",
    multiclass = "multiclass",
}

enum Regularizer {
    l2 = "l2",
    l1 = "l1",
}

enum DataFormat {
    sparse = "sparse",
    dense = "dense",
}

struct Options {
    string data; // input data path
    uint passes = 0; // number of (equivalent) passes during optimization
                     // if default = 0, use duality gap criterion instead
    ushort bits = 20; // number of bits for hashing trick
    LossType loss = LossType.squared;
    float lambda = 1.0;
    string test; // test data path
    string model_out; // path where to dump the model learnt
    string model_in; // path from where to load the model
    DataFormat data_format;
    uint n_classes; // number of classes for multiclass

    bool server_mod = false; // sqpug in prediction server mod?
    ushort server_port = 7654;
}

// Simplified version of D source code float parser, specific to char[].
// Doesn't handle Inf numbers nor Nan, and doesn't throw exceptions.
// But it's fast!
// Method under Boost Licence 1.0 at http://www.boost.org/LICENSE_1_0.txt
float to_float(char[] p) pure
{
    static immutable real[14] negtab =
        [ 1e-4096L,1e-2048L,1e-1024L,1e-512L,1e-256L,1e-128L,1e-64L,1e-32L,
                1e-16L,1e-8L,1e-4L,1e-2L,1e-1L,1.0L ];
    static immutable real[13] postab =
        [ 1e+4096L,1e+2048L,1e+1024L,1e+512L,1e+256L,1e+128L,1e+64L,1e+32L,
                1e+16L,1e+8L,1e+4L,1e+2L,1e+1L ];

    int ind = 0;
    ulong len = p.length;

    float ldval = 0.0;
    char dot = 0;
    int exp = 0;
    long msdec = 0, lsdec = 0;
    ulong msscale = 1;

    char sign = 0;
    switch (p[ind])
    {
        case '-':
            sign++;
            ind++;
            break;
        case '+':
            ind++;
            break;
        default: {}
    }

    while (ind < len)
    {
        int i = p[ind];
        while (isDigit(i))
        {
            if (msdec < (0x7FFFFFFFFFFFL-10)/10)
                msdec = msdec * 10 + (i - '0');
            else if (msscale < (0xFFFFFFFF-10)/10)
            {
                lsdec = lsdec * 10 + (i - '0');
                msscale *= 10;
            }
            else
                exp++;

            exp -= dot;
            ind++;
            if (ind == len)
                break;
            i = p[ind];
        }
        if (i == '.' && !dot)
        {
            ind++;
            dot++;
        }
        else
            break;
    }
    if (ind < len && (p[ind] == 'e' || p[ind] == 'E'))
    {
        char sexp;
        int e;

        sexp = 0;
        ind++;
        switch (p[ind])
        {
            case '-':    sexp++;
                         goto case;
            case '+':    ind++;
                         break;
            default: {}
        }
        e = 0;
        while (ind < len && isDigit(p[ind]))
        {
            if (e < 0x7FFFFFFF / 10 - 10)   // prevent integer overflow
            {
                e = e * 10 + p[ind] - '0';
            }
            ind++;
        }
        exp += (sexp) ? -e : e;
    }

    ldval = msdec;
    if (msscale != 1)
        ldval = ldval * msscale + lsdec;
    if (ldval)
    {
        uint u = 0;
        int pow = 4096;

        while (exp > 0)
        {
            while (exp >= pow)
            {
                ldval *= postab[u];
                exp -= pow;
            }
            pow >>= 1;
            u++;
        }
        while (exp < 0)
        {
            while (exp <= -pow)
            {
                ldval *= negtab[u];
                exp += pow;
            }
            pow >>= 1;
            u++;
        }
    }
    return (sign) ? -ldval : ldval;
}
