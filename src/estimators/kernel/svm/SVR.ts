import SVM from 'libsvm-js/asm';
import { Matrix } from 'ml-matrix';

import { Estimator } from '../../Estimator';

export class SVR implements Estimator {
  public model: any;
  public fitted: boolean;
  public constructor() {
    this.model = undefined;
    this.fitted = false;
  }

  public async fit(X: Matrix, y: Matrix) {
    this.model = new SVM({ type: SVM.SVM_TYPES.EPSILON_SVR });
    const prom = new Promise((resolve) => {
      resolve(this.model.train(X.to2DArray(), y.to1DArray()));
    });
    this.fitted = true;
    return prom;
  }

  public predict(X: Matrix) {
    if (this.model === undefined) {
      throw new Error('Model not fitted');
    }

    let pred = this.model.predict(X.to2DArray());
    return Matrix.from1DArray(pred.length, 1, pred);
  }
}
