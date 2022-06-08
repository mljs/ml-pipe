import { argSort, repeat, randomGaussianMatrix, Counter } from '../array';

describe('test argsort', () => {
  it('test argsort - default ascending', () => {
    const array = [1, 3, 2, 4];
    const sorted = argSort(array);
    expect(sorted).toStrictEqual([0, 2, 1, 3]);
  });

  it('test argsort - descending', () => {
    const array = [1, 3, 2, 4];
    const sorted = argSort(array, false);
    expect(sorted).toStrictEqual([3, 1, 2, 0]);
  });
});

describe('test repeat', () => {
  it('repeat', () => {
    const array = [1, 2, 3];
    const repeated = repeat(array, [2, 2, 1]);
    expect(repeated).toStrictEqual([1, 1, 2, 2, 3]);
  });

  it('repeat - not really', () => {
    const array = [1, 2, 3];
    const repeated = repeat(array, [1, 1, 1]);
    expect(repeated).toStrictEqual([1, 2, 3]);
  });
});

describe('test random gaussian matrix', () => {
  it('make sure we get approx zero mean', () => {
    const matrix = randomGaussianMatrix(10, 10);
    expect(matrix.rows).toBe(10);
    expect(matrix.columns).toBe(10);
    const mean = matrix.mean();
    expect(mean).toBeCloseTo(0, 0.1);
  });
});

describe('test getCounter', () => {
  it('getCounter', () => {
    const array = [1, 2, 3, 1, 2, 3, 1, 2, 3];
    const counter = new Counter(array);
    expect(counter.get(1)).toBe(3);
    expect([...counter.keys()]).toStrictEqual([1, 2, 3]);
  });
});
