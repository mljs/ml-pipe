import SVM from 'libsvm-js/asm';
import { Matrix } from 'ml-matrix';

import { Estimator } from '../../Estimator';

export interface SVROptions {
  kernel?: 'rbf' | 'linear' | 'polynomial' | 'sigmoid';
  // Type of SVR to perform. Epsilon and nu a different formulations of the penalty, see https://wiki.eigenvector.com/index.php?title=Svm
  type?: 'nu' | 'epsilon';
  // Degree of the polynomial kernel function (only for polynomial kernel).
  degree?: number;
  // Gamma parameter (only for rbf and poly kernels).
  gamma?: number;
  // Coefficient in kernel function (only for rbf and poly kernels).
  coef0?: number;
  // Nu parameter in loss function of Nu SVR (only for nu SVR).
  nu?: number;
  // Epsilon parameter in loss function of Epsilon SVR (only for epsilon SVR).
  epsilon?: number;
  // Value(s) to use for LIBSVM 'c' parameter.
  cost?: number;
  // cache size in MB
  cacheSize?: number;
  // tolerance of termination criterion
  tolerance?: number;
  // whether to use the shrinking heuristics
  shrinking?: boolean;
  // whether to train a SVR model for probability estimates
  probabilityEstimates?: boolean;
  // print info during learning
  quiet?: boolean;
}

// ToDo: check that this works as expected
const reg = new FinalizationRegistry((model: any) => {
  if ('free' in model) {
    model.free();
  }
});

export class SVR implements Estimator {
  public model: any;
  public fitted: boolean;
  private options: SVROptions;
  public constructor(options: SVROptions = {}) {
    this.model = undefined;
    this.fitted = false;
    this.options = options;
    reg.register(this, this.model);
  }

  private setOptions(options: SVROptions, numFeatures: number) {
    const defaultOptions = {
      kernel: 'rbf',
      type: 'epsilon',
      degree: 3,
      gamma: 'auto',
      coef0: 0,
      nu: 0.5,
      epsilon: 0.1,
      cost: 1,
      cacheSize: 200,
      tolerance: 0.001,
      shrinking: true,
      probabilityEstimates: false,
      quiet: true,
    };

    const opt = { ...defaultOptions, ...options };

    if (opt.gamma === 'auto') {
      opt.gamma = 1 / (numFeatures + 1);
    }

    if (opt.type === 'epsilon') {
      opt.type = SVM.SVM_TYPES.EPSILON_SVR;
    } else if (opt.type === 'nu') {
      opt.type = SVM.SVM_TYPES.NU_SVR;
    }
    if (opt.kernel === 'rbf') {
      opt.kernel = SVM.KERNEL_TYPES.RBF;
    } else if (opt.kernel === 'polynomial') {
      opt.kernel = SVM.KERNEL_TYPES.POLYNOMIAL;
    } else if (opt.kernel === 'sigmoid') {
      opt.kernel = SVM.KERNEL_TYPES.SIGMOID;
    } else if (opt.kernel === 'linear') {
      opt.kernel = SVM.KERNEL_TYPES.SIGMOID;
    }

    return opt;
  }

  public async fit(X: Matrix, y: Matrix) {
    this.setOptions(this.options, X.columns);
    this.model = new SVM(this.options);
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
