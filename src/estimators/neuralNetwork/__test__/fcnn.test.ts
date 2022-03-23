import { StandardScaler } from '../../../transformers/preprocessing/standardScaler';
import { trainingSet, labels, correct } from '../../../utils/testHelpers';
import { FCNN, FCNNOptions, TrainingOptions } from '../fcnn';

const fcnnOptions: FCNNOptions = {
  inputShape: 3,
  hiddenShapes: [64, 64, 32],
  outputShape: 1,
  hiddenActivation: 'relu',
  finalActivation: 'linear',
  kernelInitializer: 'glorotUniform',
  optimizer: 'adam',
  loss: 'meanSquaredError',
  metrics: [],
};

const trainingOptions: TrainingOptions = {
  epochs: 400,
  learningRate: 0.0001,
  batchSize: 36,
  validationSplit: 0.2,
};

const xScaler = new StandardScaler();
const yScaler = new StandardScaler();
const scaledTrainingSet = xScaler.fitTransform(trainingSet);
const scaledPredictions = yScaler.fitTransform(labels);

describe('test FCNN', () => {
  it('basic fit', () => {
    const fcnn = new FCNN(fcnnOptions, trainingOptions);
    void fcnn.fit(scaledTrainingSet, scaledPredictions).then(() => {});
    const result = fcnn.predict(scaledTrainingSet).to1DArray();
    expect(result).toHaveLength(25);
    let score = correct(result, scaledPredictions) / result.length;

    expect(score).toBeGreaterThanOrEqual(0.99);
  });
});
