import { Matrix } from 'ml-matrix';

import { Counter } from '../array';
import { sample } from '../sample';

describe('sample', () => {
  it('should have the expected distribution', () => {
    let predictionMatrix = new Matrix(3000, 1);
    const result = sample([1, 1, 1], [1, 2, 3], predictionMatrix);
    expect(result.size).toBe(3000);
    const foundCounts = new Counter(result.to1DArray());
    expect(foundCounts.get(1)).toBeGreaterThan(0);
    expect(foundCounts.get(2)).toBeGreaterThan(0);
    expect(foundCounts.get(3)).toBeGreaterThan(0);
    expect(foundCounts.get(1)).toBeGreaterThan(1000 - 50);
    expect(foundCounts.get(1)).toBeLessThan(1000 + 50);
  });
});
