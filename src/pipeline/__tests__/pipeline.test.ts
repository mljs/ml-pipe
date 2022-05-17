import { LinearRegressor } from '../../estimators/linear/LinearRegressor';
import {
  StandardScaler,
  TargetStandardScaler,
} from '../../transformers/preprocessing/StandardScaler';
import { trainingSet, labels, correct } from '../../utils/testHelpers';
import { Pipeline } from '../Pipeline';

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

  it('should be able to fit -- also with two transformer', async () => {
    const pipeline = new Pipeline([
      ['transformer', new StandardScaler()],
      ['transformer', new TargetStandardScaler()],
      ['regressor', new RandomForestRegressor({})],
    ]);
    await pipeline.fit(trainingSet, labels);
    const predictions = pipeline.predict(trainingSet).to1DArray();
    const score = correct(predictions, labels) / predictions.length;
    expect(score).toBeGreaterThanOrEqual(0.8);
  });

  it('should be able to fit -- also manual transforms and NN', async () => {
    const pipeline = new Pipeline([
      // ['transformer', new StandardScaler()],
      // ['yt', new TargetStandardScaler()],
      [
        'regressor',
        new FCNN({
          inputShape: 3,
          hiddenShapes: [64, 64, 32],
          outputShape: 1,
          hiddenActivation: 'relu',
          finalActivation: 'linear',
          kernelInitializer: 'glorotUniform',
          optimizer: 'adam',
          loss: 'meanSquaredError',
          epochs: 10,
          learningRate: 0.0001,
          batchSize: 36,
          validationSplit: 0.2,
        }),
      ],
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
      ['transformer', new TargetStandardScaler()],
      ['transformer', new StandardScaler()],
      [
        'regressor',
        new FCNN({
          inputShape: 3,
          hiddenShapes: [64, 64, 32],
          outputShape: 1,
          hiddenActivation: 'relu',
          finalActivation: 'linear',
          kernelInitializer: 'glorotUniform',
          optimizer: 'adam',
          loss: 'meanSquaredError',
          epochs: 10,
          learningRate: 0.0001,
          batchSize: 36,
          validationSplit: 0.2,
        }),
      ],
    ]);

    await pipeline.fit(trainingSet, labels);

    const predictions = pipeline.predict(trainingSet).to1DArray();

    const score = correct(predictions, labels) / predictions.length;
    expect(score).toBeGreaterThanOrEqual(0.7);
  }, 60000);

  it('should be able to fit -- also with transformer and y-Transformer', () => {
    const pipeline = new Pipeline([
      ['transformer', new StandardScaler()],
      ['yTransformer', new TargetStandardScaler()],
      ['regressor', new LinearRegressor()],
    ]);
    void pipeline.fit(trainingSet, labels).then(() => {});
    const predictions = pipeline.predict(trainingSet).to1DArray();
    const score = correct(predictions, labels) / predictions.length;
    expect(score).toBeGreaterThanOrEqual(0.7);
  });
});
