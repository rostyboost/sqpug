module Common;

enum LossType {
    squared = "squared",
    logistic = "logistic"
}

struct Options {
    string data; // input data path
    ushort bits; // number of bits for hashing trick
    LossType loss;
    float lambda;
    string test; // test data path
    string model_out; // path where to dump the model learnt
    string model_in; // path from where to load the model
}
