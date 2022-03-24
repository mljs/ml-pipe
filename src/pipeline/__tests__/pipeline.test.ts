import { LinearRegressor } from '../../estimators/linear/linearRegressor';
import { StandardScaler } from '../../transformers/preprocessing/standardScaler';
import { trainingSet, labels, correct } from '../../utils/testHelpers';
import { Pipeline } from '../pipeline';

describe('test basic pipeline logic', () => {
  it('invalid pipeline', () => {
    expect(() => {
      new Pipeline([['bla', new StandardScaler()]]);
    }).toThrow('Last step should be an estimator but is not.');
  });
  it('should be ok with one regressor', () => {
    expect(() => {
      new Pipeline([['bla', new LinearRegressor()]]);
    }).toBeTruthy();
  });

  it('should be able to fit', () => {
    const pipeline = new Pipeline([['regressor', new LinearRegressor()]]);
    void pipeline.fit(trainingSet, labels).then(() => {});
    const predictions = pipeline.predict(trainingSet).to1DArray();
    const score = correct(predictions, labels) / predictions.length;
    expect(score).toBeGreaterThanOrEqual(0.7);
  });

  it('should be able to fit -- also with transformer', () => {
    const pipeline = new Pipeline([
      ['transformer', new StandardScaler()],
      ['regressor', new LinearRegressor()],
    ]);
    void pipeline.fit(trainingSet, labels).then(() => {});
    const predictions = pipeline.predict(trainingSet).to1DArray();
    const score = correct(predictions, labels) / predictions.length;
    expect(score).toBeGreaterThanOrEqual(0.7);
  });
});
