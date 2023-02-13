import { Matrix } from 'ml-matrix';

import { Estimator } from '../Estimator';

export interface DummyClassifierOptions {
  strategy?: 'random' | 'majority' | 'stratified' | 'constant';
  constantValue?: number;
}

export class DummyClassifier implements Estimator {
  public fitted: boolean;
  private options: DummyClassifierOptions;
  public constructor(options: DummyClassifierOptions) {
    const { strategy = 'random', constantValue = 0 } = options;
    this.fitted = false;
    this.options = { strategy, constantValue };
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  private _fit(X: Matrix, y: Matrix) {
    switch (this.options.strategy) {
      case 'stratified':
        // get the class distribution in y
        break;
      case 'majority':
        // get the majority class in y
        break;
      default:
        break;
    }
  }

  public async fit(X: Matrix, y: Matrix) {
    this.fitted = true;
  }

  public predict(X: Matrix) {
    if (!this.fitted) {
      throw new Error('Model not fitted');
    }
    return new Matrix(this.predict(X.to2DArray()));
  }
}
