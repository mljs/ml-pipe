import {
  sequential,
  layers,
  LayersModel,
  tensor2d,
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
  metrics: string[];
}

export interface TrainingOptions {
  epochs: number;

  learningRate: number;
  batchSize: number;
  validationSplit: number;
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
    metrics: options.metrics,
  });
  return model;
}

export class FCNN implements Estimator {
  private model: LayersModel;
  private trainingOptions: TrainingOptions;
  public constructor(
    architecture: FCNNOptions,
    trainingOptions: TrainingOptions,
  ) {
    this.model = getModel(architecture);
    this.trainingOptions = trainingOptions;
  }

  public async fit(X: Matrix, y: Matrix) {
    this.model
      .fit(
        tensor2d(X.to2DArray()),
        tensor2d(y.to2DArray()),
        this.trainingOptions,
      )
      .catch((err) => {
        console.log('error', err);
      });
  }

  public predict(X: Matrix) {
    let prediction = this.model.predict(tensor2d(X.to2DArray())) as Tensor2D;
    return new Matrix(prediction.arraySync());
  }
}
