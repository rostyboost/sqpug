module Hasher;

class Hasher {
    // MurmurHash3 was written by Austin Appleby, and is placed in the public
    // domain. The author hereby disclaims copyright to this source code.
    //
    // Original C++ source code at: https://code.google.com/p/smhasher/

    private static uint _rotl32(uint x, int r)
    {
        return (x << r) | (x >> (32 - r));
    }

    // Finalization mix - force all bits of a hash block to avalanche
    private static uint fmix32(uint h)
    {
      h ^= h >> 16;
      h *= 0x85ebca6b;
      h ^= h >> 13;
      h *= 0xc2b2ae35;
      h ^= h >> 16;

      return h;
    }

    public static uint MurmurHash3(const ref char[] key, uint seed = 42)
    {
      const ubyte * data = cast(const(ubyte*))key;
      uint len = cast(uint)key.length;
      const int nblocks = len / 4;

      uint h1 = seed;

      const uint c1 = 0xcc9e2d51;
      const uint c2 = 0x1b873593;

      // body
      const uint * blocks = cast(const (uint *))(data + nblocks*4);

      for(int i = -nblocks; i; i++)
      {
        uint k1 = blocks[i];

        k1 *= c1;
        k1 = _rotl32(k1,15);
        k1 *= c2;
        
        h1 ^= k1;
        h1 = _rotl32(h1,13); 
        h1 = h1*5+0xe6546b64;
      }

      // tail
      const ubyte * tail = cast(const (ubyte*))(data + nblocks*4);

      uint k1 = 0;

      switch(len & 3)
      {
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0];
                k1 *= c1; k1 = _rotl32(k1,15); k1 *= c2; h1 ^= k1;
        default:
            ; // so D compiler shuts up.
      };

      // finalization
      h1 ^= len;
      h1 = fmix32(h1);

      return h1;
    }
}
