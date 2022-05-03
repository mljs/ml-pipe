import { Matrix } from 'ml-matrix';

import { meanAbsoluteError, meanSquaredError } from '../regression';

describe('test meanSquaredError', () => {
  it('should return 0 for equal matrices', () => {
    const y = Matrix.from1DArray(2, 1, [1, 2]);
    const yHat = Matrix.from1DArray(2, 1, [1, 2]);
    expect(meanSquaredError(y, yHat)).toBe(0);
  });
});

describe('test meanAbsoluteError', () => {
  it('should return 0 for equal matrices', () => {
    const y = Matrix.from1DArray(2, 1, [1, 2]);
    const yHat = Matrix.from1DArray(2, 1, [1, 2]);
    expect(meanAbsoluteError(y, yHat)).toBe(0);
  });
});
