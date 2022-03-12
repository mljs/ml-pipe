import { Matrix } from 'ml-matrix';

export interface Estimator {
  fit: (X: Matrix, y: Matrix) => void;
  predict: (X: Matrix) => Promise<Matrix>;
}
