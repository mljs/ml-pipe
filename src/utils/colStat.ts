import { Matrix } from 'ml-matrix';

export function columnWiseMin(matrix: Matrix): number[] {
  let mins = [];
  for (let i = 0; i < matrix.columns; i++) {
    let min = Number.POSITIVE_INFINITY;
    for (let j = 0; j < matrix.rows; j++) {
      min = Math.min(min, matrix.get(j, i));
    }
    mins.push(min);
  }
  return mins;
}

export function columnWiseMax(matrix: Matrix): number[] {
  let maxes = [];
  for (let i = 0; i < matrix.columns; i++) {
    let max = Number.NEGATIVE_INFINITY;
    for (let j = 0; j < matrix.rows; j++) {
      max = Math.max(max, matrix.get(j, i));
    }
    maxes.push(max);
  }
  return maxes;
}
