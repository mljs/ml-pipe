import {
  sequential,
  layers,
  LayersModel,
  tensor2d,
  tensor1d,
  Tensor2D,
} from '@tensorflow/tfjs';
import { Matrix } from 'ml-matrix';

import { Estimator } from '../estimator';

export interface FCNNOptions {
  inputShape: number;
  hiddenShapes: number[];
  outputShape: number;
  hiddenActivation: 'relu' | 'sigmoid' | 'tanh' | 'linear';
  finalActivation: 'relu' | 'sigmoid' | 'tanh' | 'linear';
  kernelInitializer:
    | 'leCunNormal'
    | 'glorotUniform'
    | 'randomUniform'
    | 'truncatedNormal'
    | 'varianceScaling';
  optimizer: 'sgd' | 'rmsprop' | 'adam' | 'adagrad' | 'adadelta';
  loss: 'meanSquaredError' | 'binaryCrossentropy' | 'categoricalCrossentropy';
  epochs: number;
  learningRate: number;
  batchSize: number;
  validationSplit: number;
}

function validateOptions(options: FCNNOptions) {
  let {
    inputShape,
    hiddenShapes = [32, 32],
    outputShape = 1,
    hiddenActivation = 'relu',
    kernelInitializer = 'glorotUniform',
    optimizer = 'adam',
    loss = 'meanSquaredError',
    epochs = 200,
    learningRate = 0.001,
    batchSize = 32,
    validationSplit = 0.2,
    finalActivation = 'linear',
  } = options;
  if (!(typeof inputShape === 'number')) {
    throw new Error('inputShape must be a number');
  }
  if (inputShape <= 0) {
    throw new Error('inputShape must be greater than 0');
  }

  if (!(validationSplit < 1 && validationSplit > 0)) {
    throw new Error('validationSplit must be between 0 and 1');
  }

  if (!(batchSize > 0)) {
    throw new Error('batchSize must be greater than 0');
  }

  if (!(epochs > 0)) {
    throw new Error('epochs must be greater than 0');
  }

  if (!(learningRate > 0)) {
    throw new Error('learningRate must be greater than 0');
  }

  return {
    inputShape: inputShape,
    hiddenShapes: hiddenShapes,
    outputShape: outputShape,
    hiddenActivation: hiddenActivation,
    kernelInitializer: kernelInitializer,
    optimizer: optimizer,
    loss: loss,
    epochs: epochs,
    learningRate: learningRate,
    batchSize: batchSize,
    validationSplit: validationSplit,
    finalActivation: finalActivation,
  };
}

function getModel(options: FCNNOptions): LayersModel {
  const model = sequential();
  model.add(
    layers.dense({
      inputShape: [options.inputShape],
      units: options.hiddenShapes[0],
      activation: options.hiddenActivation,
      kernelInitializer: options.kernelInitializer,
    }),
  );
  for (let i = 1; i < options.hiddenShapes.length; i++) {
    model.add(
      layers.dense({
        units: options.hiddenShapes[i],
        activation: options.hiddenActivation,
        kernelInitializer: options.kernelInitializer,
      }),
    );
  }
  model.add(
    layers.dense({
      units: options.outputShape,
      activation: options.finalActivation,
      kernelInitializer: options.kernelInitializer,
    }),
  );
  model.compile({
    optimizer: options.optimizer,
    loss: options.loss,
  });
  return model;
}

export class FCNN implements Estimator {
  private model: LayersModel;
  private options: FCNNOptions;
  public constructor(options: FCNNOptions) {
    options = validateOptions(options);
    this.model = getModel(options);
    this.options = options;
  }

  public async fit(X: Matrix, y: Matrix) {
    this.model
      .fit(tensor2d(X.to2DArray()), tensor1d(y.to1DArray()), this.options)
      .catch((err) => {
        throw new Error(err.message);
      });
  }

  public predict(X: Matrix) {
    let prediction = this.model.predict(tensor2d(X.to2DArray())) as Tensor2D;
    return new Matrix(prediction.arraySync());
  }
}
