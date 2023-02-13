import Matrix from 'ml-matrix';

export function sample(
  counts: Array<number>,
  classNames: Array<number>,
  predictionMatrix: Matrix,
) {
  // fill prediction matrix with samples
  // from the `classNames` such that the resulting
  // matrix has the same distribution as the `counts`
  // array
  const total = counts.reduce((a, b) => a + b, 0);
  const probabilities = counts.map((count) => count / total);
  const cumulativeProbabilities = probabilities.reduce(
    (acc, p) => [...acc, acc[acc.length - 1] + p],
    [0],
  );
  const randomNumbers = new Array(predictionMatrix.rows)
    .fill(0)
    .map(() => Math.random());
  for (let i = 0; i < predictionMatrix.rows; i++) {
    const randomNumber = randomNumbers[i];
    const classIndex =
      cumulativeProbabilities.findIndex((p) => randomNumber < p) - 1;
    predictionMatrix.set(i, 0, classNames[classIndex]);
  }
  return predictionMatrix;
}
