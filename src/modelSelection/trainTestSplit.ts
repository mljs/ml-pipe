import { Matrix } from 'ml-matrix';

import { argSort, cumSum, repeat } from '../utils/array';
import { clamp } from '../utils/number';

export function trainTestSplit() {}

function shuffleArray(x: Array<any>) {
  // Durstenfeld shuffle
  const array = [...x];
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

function stratifiedTrainTestSplit() {}

export function generatedShuffledIndices(x: Array<any> | Matrix) {
  const shuffled = shuffleArray(rangeFromArrayMatrix(x));
  return shuffled;
}

function rangeFromArrayMatrix(x: Array<any> | Matrix) {
  let range;
  if (Array.isArray(x)) {
    range = [...Array(x.length).keys()];
  } else {
    range = [...Array(x.rows).keys()];
  }
  return range;
}

export function getStratificationIndices(
  stratifyArray: Array<any>,
  trainTestRatio: number,
) {
  // todo:  complain if there is class with less than 2 points
  // todo: also complain if there is only one class
  // Stratified split implementation inspired by https://stackoverflow.com/questions/15838733/stratified-sampling-in-numpy
  // First we need to find the unique values in the array as well as their indices and counts
  const { indices, counts } = uniqueValuesIndicesAndCounts(stratifyArray);

  const blockCounts: Array<number> = Object.values(counts);

  const blocks = argSort(indices);
  // for example, [0, 2, 5, 10]
  const blockStartingIndices = [0].concat(cumSum(blockCounts));

  let fractions = blockCounts.map((count: number) =>
    expectedFraction(count, trainTestRatio),
  );

  let thresholds = repeat(fractions, blockCounts);

  let setNext = false;
  for (let i = 0; i < thresholds.length; i++) {
    if (blockStartingIndices.includes(i)) {
      thresholds[i] = 1;
      setNext = true;
    }
    if (setNext) {
      thresholds[i + 1] = 0;
      setNext = false;
    } else {
      thresholds[i] = Number(thresholds[i] > Math.random());
    }
  }
  let aIndices: Array<number> = [];
  let bIndices: Array<number> = [];
  for (let i = 0; i < indices.length; i++) {
    if (thresholds[i] === 1) {
      aIndices.push(blocks[i]);
    } else {
      bIndices.push(blocks[i]);
    }
  }

  return { aIndices, bIndices };
}

function uniqueValuesIndicesAndCounts(x: Array<any>) {
  const uniqueValues = [...new Set(x)];
  const range = rangeFromArrayMatrix(uniqueValues);
  const uniqueIndices = Object.fromEntries(
    uniqueValues.map((e, i) => [e, range[i]]),
  );
  let counts = Object.fromEntries(uniqueValues.map((e) => [e, 0]));
  const indices = [];
  for (let val of x) {
    indices.push(uniqueIndices[val]);
    counts[val] += 1;
  }
  return { uniqueValues, indices, counts };
}

function expectedFraction(numPoints: number, trainTestRatio: number) {
  if (trainTestRatio < 1) {
    console.warn('trainTestRatio typically greater than 1');
  }
  return clamp(
    ((numPoints - 1) * trainTestRatio - 1) /
      ((numPoints - 2) * (trainTestRatio + 1)),
    0,
    1,
  );
}

export const testables = { expectedFraction, uniqueValuesIndicesAndCounts };
