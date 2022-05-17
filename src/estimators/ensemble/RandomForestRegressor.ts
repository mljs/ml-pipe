import { Matrix } from 'ml-matrix';
import { RandomForestRegression } from 'ml-random-forest';

import { Estimator } from '../Estimator';

export class RandomForestRegressor implements Estimator {
  private options: any;
  public fitted: boolean;
  private regressor: RandomForestRegression;
  public constructor(options: any = {}) {
    this.options = options;
    this.regressor = new RandomForestRegression(options);
    this.fitted = false;
  }

  public async fit(X: Matrix, y: Matrix) {
    const prom = new Promise((resolve) => {
      resolve(this.regressor.train(X.to2DArray(), y.to1DArray()));
    });
    this.fitted = true;
    return prom;
  }

  public predict(X: Matrix) {
    return Matrix.from1DArray(X.rows, 1, this.regressor.predict(X.to2DArray()));
  }
}
