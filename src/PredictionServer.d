module PredictionServer;

import std.conv;
import std.socket;
import std.stdio;

import Common;
import Hasher;
import Learner;


class PredictionServer {

    ushort _port;
    Socket _server;
    Learner _learner;
    uint _bitMask;

    this(const ushort port, ref Learner learner)
    {
        _learner = learner;
        _bitMask = (1 << learner.bits) - 1;
        _port = port;
        _server = new TcpSocket();
        _server.setOption(SocketOptionLevel.SOCKET,
                          SocketOption.REUSEADDR,
                          true);
        _server.bind(new InternetAddress(_port));
        _server.listen(1);
    }

    void _parse_input(char[] buffer, long max_ind,
                      ref Feature[] features)
    {
        int feat_start = 0;
        int feat_end = 0;
        uint feat_hash = -1;

        int val_start = 0;
        int val_end = 0;
        float feat_val = -1;

        int ind = 0;
        while(ind < max_ind)
        {
            switch(buffer[ind])
            {
                case ':':
                    feat_end = ind;
                    val_start = ind+1;
                    feat_hash = Hasher.Hasher.MurmurHash3(
                        buffer[feat_start..feat_end]) & _bitMask;
                    break;
                case ' ':
                    val_end = ind;
                    feat_start = ind + 1;
                    feat_val = to_float(buffer[val_start..val_end]);
                    features ~= Feature(feat_hash, feat_val);
                    break;
                case '\n': // eol, need to dump last feature
                    goto case ' ';
                default: // regular character
                    break;
            }
            ind++;
        }
    }

    void serve_forever()
    {
        while(true)
        {
            Socket client = _server.accept();
            char[4096] buffer;
            auto ind_end = client.receive(buffer);

            Feature[] features;
            _parse_input(buffer, ind_end, features);

            float pred = _learner.predict(features);

            client.send(to!string(pred));

            client.shutdown(SocketShutdown.BOTH);
            client.close();
        }
    }

}
