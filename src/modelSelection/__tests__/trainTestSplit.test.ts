import { testables } from '../trainTestSplit';

const { expectedFraction, uniqueValuesIndicesAndCounts } = testables;

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

describe('test unique valuesIndicesAndCounts', () => {
  it('constant array', () => {
    let { uniqueValues, indices, counts } = uniqueValuesIndicesAndCounts([
      1, 1, 1, 1,
    ]);
    expect(uniqueValues).toStrictEqual([1]);
    expect(indices).toStrictEqual([0, 0, 0, 0]);
    expect(counts).toStrictEqual({ 1: 4 });
  });

  it('two kinds array', () => {
    let { uniqueValues, indices, counts } = uniqueValuesIndicesAndCounts([
      1, 2, 1, 1,
    ]);
    expect(uniqueValues).toStrictEqual([1, 2]);
    expect(indices).toStrictEqual([0, 1, 0, 0]);
    expect(counts).toStrictEqual({ 1: 3, 2: 1 });
  });
});
