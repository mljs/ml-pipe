import { Matrix } from 'ml-matrix';

export function approx(val: number, expected: number, eps: number): boolean {
  return val - eps < expected && expected < val + eps;
}

let dataset = [
  [73, 80, 75, 152],
  [93, 88, 93, 185],
  [89, 91, 90, 180],
  [96, 98, 100, 196],
  [73, 66, 70, 142],
  [53, 46, 55, 101],
  [69, 74, 77, 149],
  [47, 56, 60, 115],
  [87, 79, 90, 175],
  [79, 70, 88, 164],
  [69, 70, 73, 141],
  [70, 65, 74, 141],
  [93, 95, 91, 184],
  [79, 80, 73, 152],
  [70, 73, 78, 148],
  [93, 89, 96, 192],
  [78, 75, 68, 147],
  [81, 90, 93, 183],
  [88, 92, 86, 177],
  [78, 83, 77, 159],
  [82, 86, 90, 177],
  [86, 82, 89, 175],
  [78, 83, 85, 175],
  [76, 83, 71, 149],
  [96, 93, 95, 192],
];

export let trainingSet = new Matrix(dataset.length, 3);
export let labels = new Matrix(dataset.length, 1);

for (let i = 0; i < dataset.length; ++i) {
  trainingSet.setRow(i, dataset[i].slice(0, 3));
  labels.setRow(i, [dataset[i][3]]);
}

export function correct(result: Array<number>, predictions: Matrix): number {
  return result.reduce((prev: number, value: number, index: number) => {
    return approx(value, predictions.get(index, 0), 10) ? prev + 1 : prev;
  }, 0);
}
