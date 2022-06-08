// leave only index
import { Matrix } from 'ml-matrix';

import { gaussRandom } from './number';

const decorateWIndex = (v: number, i: number) => [v, i];
const removeDecoration = (a: Array<any>) => a[1];

export function argSort(
  arr: Array<number>,
  ascending?: boolean,
): Array<number> {
  const sortAscending = ascending === undefined ? true : ascending;
  let sorted = arr
    .map(decorateWIndex)
    .sort((a, b) => a[0] - b[0])
    .map(removeDecoration);
  if (sortAscending) {
    return sorted;
  } else {
    return sorted.reverse();
  }
}

export function cumSum(arr: Array<number>): Array<number> {
  const cumulativeSum = (
    (sum: number) => (value: number) =>
      (sum += value)
  )(0);
  return arr.map(cumulativeSum);
}

export function repeat(arr: Array<any>, times: Array<number>): Array<any> {
  return arr.flatMap((e, index) => Array(times[index]).fill(e));
}

export function randomGaussianMatrix(rows: number, cols: number): Matrix {
  let arr = Array(rows * cols)
    .fill(0)
    .map(() => gaussRandom());
  return Matrix.from1DArray(rows, cols, arr);
}

export class Counter<CounterKey> extends Map<CounterKey, number> {
  constructor(items: Iterable<CounterKey>) {
    super();

    for (let it of items) {
      this.add(it);
    }
  }

  add(it: CounterKey) {
    this.set(it, (this.get(it) || 0) + 1);
  }
}
