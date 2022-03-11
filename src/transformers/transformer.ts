import { Matrix } from 'ml-matrix';

export interface Transformer {
  fit: (X: Matrix) => void;
  transform: (X: Matrix) => Matrix;
  inverseTransform: (X: Matrix) => Matrix;
}
