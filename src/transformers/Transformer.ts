import { Matrix } from 'ml-matrix';

export interface Transformer {
  onTarget: boolean;
  fitted: boolean;
  fit: (X: Matrix) => void;
  transform: (X: Matrix) => Matrix;
  inverseTransform: (X: Matrix) => Matrix;
  fitTransform: (X: Matrix) => Matrix;
}
