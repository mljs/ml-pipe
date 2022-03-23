import { trainingSet, labels, correct } from '../../../utils/testHelpers';
import { LinearRegressor } from '../linearRegressor';

describe('test linear regressor', () => {
  it('on test data', () => {
    const regressor = new LinearRegressor();
    void regressor.fit(trainingSet, labels).then(() => {});
    const result = regressor.predict(trainingSet).to1DArray();
    expect(result).toHaveLength(25);
    let score = correct(result, labels) / result.length;

    expect(score).toBeGreaterThanOrEqual(0.99);
  });
});
