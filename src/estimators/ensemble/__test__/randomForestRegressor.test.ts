import { trainingSet, labels, correct } from '../../../utils/testHelpers';
import { RandomForestRegressor } from '../randomForestRegressor';

// Test taken from the ml-rf package

let options = {
  seed: 3,
  maxFeatures: 2,
  replacement: false,
  nEstimators: 200,
  treeOptions: undefined,
  useSampleBagging: true,
};

let regression = new RandomForestRegressor(options);
void regression.fit(trainingSet, labels).then(() => {});
let result = regression.predict(trainingSet).to1DArray();
describe('Random Forest Regression', () => {
  it('Random Forest regression with scores psychology from Houghton Mifflin', () => {
    let score = correct(result, labels) / result.length;

    expect(score).toBeGreaterThanOrEqual(0.7);
  });
});
