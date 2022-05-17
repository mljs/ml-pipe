import { Matrix } from 'ml-matrix';

export interface Estimator {
  fitted: boolean;
  model?: any;
  fit: (X: Matrix, y: Matrix) => Promise<any>;
  predict: (X: Matrix) => Matrix;
}
