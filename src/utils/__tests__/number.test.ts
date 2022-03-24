import { clamp } from '../number';

describe('test clamp', () => {
  it('infy', () => {
    expect(clamp(Infinity, 0, 1)).toBe(1);
  });
  it('negative', () => {
    expect(clamp(-1, 0, 1)).toBe(0);
  });
});
