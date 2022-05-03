import { Matrix } from 'ml-matrix';

export function meanSquaredError(y: Matrix, yHat: Matrix): number {
  let sum = 0;

  for (let i = 0; i < y.rows; i++) {
    for (let j = 0; j < y.columns; j++) {
      sum += Math.pow(y.get(i, j) - yHat.get(i, j), 2);
    }
  }

  return sum / (y.rows * y.columns);
}

export function meanAbsoluteError(y: Matrix, yHat: Matrix): number {
  let sum = 0;

  for (let i = 0; i < y.rows; i++) {
    for (let j = 0; j < y.columns; j++) {
      sum += Math.abs(y.get(i, j) - yHat.get(i, j));
    }
  }

  return sum / (y.rows * y.columns);
}
