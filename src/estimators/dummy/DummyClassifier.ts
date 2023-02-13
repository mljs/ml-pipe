import { Matrix } from 'ml-matrix';
import { argmax } from 'ml-spectra-processing';

import { Counter } from '../../utils/array';
import { sample } from '../../utils/sample';
import { Estimator } from '../Estimator';

export interface DummyClassifierOptions {
  strategy?: 'random' | 'majority' | 'stratified' | 'constant';
  constantValue?: number;
}

export class DummyClassifier implements Estimator {
  public fitted: boolean;
  private options: DummyClassifierOptions;
  private classStats: Counter<any>;
  public constructor(options: DummyClassifierOptions) {
    const { strategy = 'random', constantValue = 0 } = options;
    this.fitted = false;
    this.options = { strategy, constantValue };
    this.classStats = new Counter([]);
  }

  private _fit(y: Matrix) {
    this.classStats = new Counter(y.to1DArray());
  }

  private _predict(X: Matrix) {
    const keys = [...this.classStats.keys()];
    const counts = keys.map((key) => this.classStats.get(key) || 0);
    const maxClassIndex = argmax(counts);
    const maxClassName = keys[maxClassIndex];
    let prediction = new Matrix(X.rows, 1);
    switch (this.options.strategy) {
      case 'stratified':
        prediction = sample(counts, keys, prediction);
        break;
      case 'majority':
        for (let i = 0; i < X.rows; i++) {
          prediction.set(i, 0, maxClassName);
        }
        break;
      default:
        break;
    }
    return prediction;
  }

  public async fit(X: Matrix, y: Matrix) {
    this._fit(y);
    this.fitted = true;
  }

  public predict(X: Matrix) {
    if (!this.fitted) {
      throw new Error('Model not fitted');
    }
    return new Matrix(this._predict(X));
  }
}
