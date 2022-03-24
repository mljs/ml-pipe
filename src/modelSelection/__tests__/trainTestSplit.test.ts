import { testables } from '../trainTestSplit';

const { expectedFraction } = testables;

describe('test expected fraction', () => {
  it('three 50/50 case', () => {
    // have three points and 50/50 split
    expect(expectedFraction(3, 2)).toBe(1);
  });
  it('two 50/50 case', () => {
    // have two points and 50/50 split
    expect(expectedFraction(2, 2)).toBe(1);
  });
});
