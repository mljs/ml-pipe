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
  // The number of input features */
  inputShape: number;
  // Length of array = number of hidden layers, each element is the number of neurons in that layer */
  hiddenShapes: number[];
  // The number of outputs/targets. */
  outputShape: number;
  // The activation function for the hidden layers. */
  hiddenActivation: 'relu' | 'sigmoid' | 'tanh' | 'linear';
  // The activation function for the output layer. */
  finalActivation: 'relu' | 'sigmoid' | 'tanh' | 'linear';
  // The initializer for the kernel weights. */
  kernelInitializer:
    | 'leCunNormal'
    | 'glorotUniform'
    | 'randomUniform'
    | 'truncatedNormal'
    | 'varianceScaling';
  // The optimizer to use. */
  optimizer: 'sgd' | 'rmsprop' | 'adam' | 'adagrad' | 'adadelta';
  // The loss function to use. */
  loss: 'meanSquaredError' | 'binaryCrossentropy' | 'categoricalCrossentropy';
  // The number of epochs to train the model. */
  epochs: number;
  // The learning rate for the optimizer. */
  learningRate: number;
  // The batch size for the optimizer. */
  batchSize: number;
  // The validation split for the model. (Can be used for early stopping) */
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
  if (typeof inputShape !== 'number') {
    throw new Error('inputShape must be a number');
  }
  if (inputShape <= 0) {
    throw new Error('inputShape must be greater than 0');
  }

  if (validationSplit <= 0 || validationSplit >= 1) {
    throw new Error('validationSplit must be between 0 and 1');
  }

  if (batchSize <= 0) {
    throw new Error('batchSize must be greater than 0');
  }

  if (epochs <= 0) {
    throw new Error('epochs must be greater than 0');
  }

  if (learningRate <= 0) {
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

/**
 * Implementation of a fully connected neural network using Tensorflow.js.
 *
 * @export
 * @class FCNN
 * @implements {Estimator}
 */
export class FCNN implements Estimator {
  private model: LayersModel;
  private options: FCNNOptions;

  /**
   * Creates an instance of FCNN.
   * @param {FCNNOptions} options
   * @memberof FCNN
   */
  public constructor(options: FCNNOptions) {
    options = validateOptions(options);
    this.model = getModel(options);
    this.options = options;
  }
  /**
   * Trains the model.
   *
   * @param {Matrix} X - The feature matrix.
   * @param {Matrix} y - The target matrix.
   * @memberof FCNN
   */
  public async fit(X: Matrix, y: Matrix) {
    this.model
      .fit(tensor2d(X.to2DArray()), tensor1d(y.to1DArray()), this.options)
      .catch((err) => {
        throw new Error(err.message);
      });
  }
  /**
   * Predicts the output for a given feature matrix.
   *
   * @param {Matrix} X - feature matrix
   * @return {Matrix}
   * @memberof FCNN
   */
  public predict(X: Matrix) {
    let prediction = this.model.predict(tensor2d(X.to2DArray())) as Tensor2D;
    return new Matrix(prediction.arraySync());
  }
}