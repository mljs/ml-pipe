import { StandardScaler } from '../../../../transformers/preprocessing/StandardScaler';
import { trainingSet, labels, correct } from '../../../../utils/testHelpers';
import { SVR } from '../SVR';

const xScaler = new StandardScaler();
const yScaler = new StandardScaler();
const scaledTrainingSet = xScaler.fitTransform(trainingSet);
const scaledPredictions = yScaler.fitTransform(labels);

describe('test SVR', () => {
  it('on test data', async () => {
    const regressor = new SVR();
    await regressor.fit(scaledTrainingSet, scaledPredictions).then(() => {});
    const result = regressor.predict(scaledTrainingSet).to1DArray();
    expect(result).toHaveLength(25);
    let score = correct(result, scaledPredictions) / result.length;

    expect(score).toBeGreaterThanOrEqual(0.99);
  });
});
