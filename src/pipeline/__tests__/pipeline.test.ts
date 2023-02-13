import seedrandom from 'seedrandom';

import { RandomForestRegressor } from '../../estimators/ensemble/RandomForestRegressor';
import { LinearRegressor } from '../../estimators/linear/LinearRegressor';
import { FCNN } from '../../estimators/neuralNetwork/FCNN';
import {
  StandardScaler,
  TargetStandardScaler,
} from '../../transformers/preprocessing/StandardScaler';
import { trainingSet, labels, correct } from '../../utils/testHelpers';
import { Pipeline } from '../Pipeline';

describe('test basic pipeline logic', () => {
  seedrandom('test');
  it('invalid pipeline', () => {
    expect(() => {
      new Pipeline([{ name: 'bla', object: new StandardScaler() }]);
    }).toThrow('Last step should be an estimator but is not.');
  });
  it('should be ok with one regressor', () => {
    expect(() => {
      new Pipeline([{ name: 'bla', object: new LinearRegressor() }]);
    }).toBeTruthy();
  });

  it('should be able to fit', () => {
    const pipeline = new Pipeline([
      { name: 'regressor', object: new LinearRegressor() },
    ]);
    void pipeline.fit(trainingSet, labels).then(() => {});
    const predictions = pipeline.predict(trainingSet).to1DArray();
    const score = correct(predictions, labels) / predictions.length;
    expect(score).toBeGreaterThanOrEqual(0.7);
  });

  it('should be able to fit -- also with transformer', () => {
    const pipeline = new Pipeline([
      { name: 'transformer', object: new StandardScaler() },
      { name: 'regressor', object: new LinearRegressor() },
    ]);
    void pipeline.fit(trainingSet, labels).then(() => {});
    const predictions = pipeline.predict(trainingSet).to1DArray();
    const score = correct(predictions, labels) / predictions.length;

    expect(score).toBeGreaterThanOrEqual(0.7);
  });

  it('should be able to fit -- also with two transformer', async () => {
    const pipeline = new Pipeline([
      { name: 'transformer', object: new StandardScaler() },
      { name: 'transformer', object: new TargetStandardScaler() },
      { name: 'regressor', object: new RandomForestRegressor({}) },
    ]);
    await pipeline.fit(trainingSet, labels);
    const predictions = pipeline.predict(trainingSet).to1DArray();
    const score = correct(predictions, labels) / predictions.length;
    expect(score).toBeGreaterThanOrEqual(0.8);
  });

  it('should be able to fit -- also manual transforms and NN', async () => {
    const pipeline = new Pipeline([
      {
        name: 'regressor',
        object: new FCNN({
          hiddenShapes: [64, 64, 32],
          hiddenActivation: 'relu',
          finalActivation: 'linear',
          kernelInitializer: 'glorotUniform',
          optimizer: 'adam',
          loss: 'meanSquaredError',
          epochs: 20,
          learningRate: 0.0001,
          batchSize: 36,
          validationSplit: 0.2,
        }),
      },
    ]);
    const xScaler = new StandardScaler();
    const yScaler = new StandardScaler();
    const scaledTrainingSet = xScaler.fitTransform(trainingSet);
    const scaledPredictions = yScaler.fitTransform(labels);

    await pipeline.fit(scaledTrainingSet, scaledPredictions).then(() => {});
    const predictions = pipeline.predict(scaledTrainingSet).to1DArray();
    const score = correct(predictions, scaledPredictions) / predictions.length;
    expect(score).toBeGreaterThanOrEqual(0.7);
  }, 60000);

  it('should be able to fit -- also with transformer and NN', async () => {
    const pipeline = new Pipeline([
      { name: 'transformer', object: new TargetStandardScaler() },
      { name: 'transformer', object: new StandardScaler() },
      {
        name: 'regressor',
        object: new FCNN({
          hiddenShapes: [64, 64, 32],
          hiddenActivation: 'relu',
          finalActivation: 'linear',
          kernelInitializer: 'glorotUniform',
          optimizer: 'adam',
          loss: 'meanSquaredError',
          epochs: 20,
          learningRate: 0.0001,
          batchSize: 36,
          validationSplit: 0.2,
        }),
      },
    ]);

    await pipeline.fit(trainingSet, labels);

    const predictions = pipeline.predict(trainingSet).to1DArray();

    const score = correct(predictions, labels) / predictions.length;
    expect(score).toBeGreaterThanOrEqual(0.7);
  }, 60000);

  it('should be able to fit -- also with transformer and y-Transformer', () => {
    const pipeline = new Pipeline([
      { name: 'transformer', object: new StandardScaler() },
      { name: 'yTransformer', object: new TargetStandardScaler() },
      { name: 'regressor', object: new LinearRegressor() },
    ]);
    void pipeline.fit(trainingSet, labels).then(() => {});
    const predictions = pipeline.predict(trainingSet).to1DArray();
    const score = correct(predictions, labels) / predictions.length;
    expect(score).toBeGreaterThanOrEqual(0.7);
  });
});
