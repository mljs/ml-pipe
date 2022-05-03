import { Matrix } from 'ml-matrix';

import { LinearRegressor } from '../../estimators/linear/LinearRegressor';
import { trainingSet, labels } from '../../utils/testHelpers';
import { meanAbsoluteError, meanSquaredError, doFAIC } from '../regression';

describe('test meanSquaredError', () => {
  it('should return 0 for equal matrices', () => {
    const y = Matrix.from1DArray(2, 1, [1, 2]);
    const yHat = Matrix.from1DArray(2, 1, [1, 2]);
    expect(meanSquaredError(y, yHat)).toBe(0);
  });

  it('should return 1 for different matrices', () => {
    const y = Matrix.from1DArray(2, 1, [1, 2]);
    const yHat = Matrix.from1DArray(2, 1, [2, 3]);
    expect(meanSquaredError(y, yHat)).toBe(1);
  });
});

describe('test meanAbsoluteError', () => {
  it('should return 0 for equal matrices', () => {
    const y = Matrix.from1DArray(2, 1, [1, 2]);
    const yHat = Matrix.from1DArray(2, 1, [1, 2]);
    expect(meanAbsoluteError(y, yHat)).toBe(0);
  });

  it('should return 1 for different matrices', () => {
    const y = Matrix.from1DArray(2, 1, [1, 2]);
    const yHat = Matrix.from1DArray(2, 1, [2, 3]);
    expect(meanAbsoluteError(y, yHat)).toBe(1);
  });
});

describe('test doFAIC', () => {
  it('linear model - make sure the call to the function works', async () => {
    const regressor = new LinearRegressor();
    await regressor.fit(trainingSet, labels);
    const faic = await doFAIC(trainingSet, labels, regressor);
    expect(faic).toBeGreaterThanOrEqual(2);
  });
});
