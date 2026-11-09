/** Integer-seeded MT19937 compatible with Python random.Random (world version v1). */
export class Random {
  private state = new Uint32Array(624);
  private index = 624;
  constructor(seed: number) {
    if (!Number.isSafeInteger(seed)) throw new Error('Seed must be a safe integer');
    let value = BigInt(Math.abs(seed));
    const key: number[] = [];
    do { key.push(Number(value & 0xffffffffn)); value >>= 32n; } while (value);
    this.state[0] = 19650218;
    for (let i = 1; i < 624; i++) this.state[i] = (Math.imul(1812433253, this.state[i - 1] ^ (this.state[i - 1] >>> 30)) + i) >>> 0;
    let i = 1, j = 0;
    for (let k = Math.max(624, key.length); k; k--) {
      this.state[i] = ((this.state[i] ^ Math.imul(this.state[i - 1] ^ (this.state[i - 1] >>> 30), 1664525)) + key[j] + j) >>> 0;
      if (++i >= 624) { this.state[0] = this.state[623]; i = 1; }
      if (++j >= key.length) j = 0;
    }
    for (let k = 623; k; k--) {
      this.state[i] = ((this.state[i] ^ Math.imul(this.state[i - 1] ^ (this.state[i - 1] >>> 30), 1566083941)) - i) >>> 0;
      if (++i >= 624) { this.state[0] = this.state[623]; i = 1; }
    }
    this.state[0] = 0x80000000;
  }
  private uint32() {
    if (this.index >= 624) {
      for (let i = 0; i < 624; i++) {
        const y = (this.state[i] & 0x80000000) | (this.state[(i + 1) % 624] & 0x7fffffff);
        this.state[i] = this.state[(i + 397) % 624] ^ (y >>> 1) ^ ((y & 1) ? 0x9908b0df : 0);
      }
      this.index = 0;
    }
    let y = this.state[this.index++];
    y ^= y >>> 11; y ^= (y << 7) & 0x9d2c5680; y ^= (y << 15) & 0xefc60000; y ^= y >>> 18;
    return y >>> 0;
  }
  random() { return ((this.uint32() >>> 5) * 67108864 + (this.uint32() >>> 6)) / 9007199254740992; }
  below(n: number) {
    const bits = Math.floor(Math.log2(n)) + 1;
    let result: number;
    do { result = this.uint32() >>> (32 - bits); } while (result >= n);
    return result;
  }
  choice<T>(values: T[]) { return values[this.below(values.length)]; }
}
