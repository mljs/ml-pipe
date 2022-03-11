import { Matrix } from 'ml-matrix';

import { Transformer } from './transformer';

export class StandardScaler implements Transformer {
  private std: Matrix;
  private mean: Matrix;

  public constructor() {
    this.std = new Matrix([]);
    this.mean = new Matrix([]);
  }

  public fit(X: Matrix) {
    this.std = new Matrix([X.variance('column')]);
    this.mean = new Matrix([X.mean('column')]);
  }

  public transform(X: Matrix) {
    return X.sub(this.mean).div(this.std);
  }

  public inverseTransform(X: Matrix) {
    return X.mul(this.std).add(this.mean);
  }
}
